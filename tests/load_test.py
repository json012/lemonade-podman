#!/usr/bin/env -S uv run python
"""
Load Testing Script for LiteLLM Models
This script performs comprehensive load testing to identify lockup conditions.

Key Features:
- Rate limiting sensitive: 429 responses are detected and not retried to avoid skewing timing measurements
- Dynamic model discovery: Automatically fetches available models from /models endpoint
- Comprehensive test scenarios: Parallel calls, memory persistence, stress conditions

Environment Variables Required:
    OPENAI_API_KEY: Your API key for authentication
    OPENAI_BASE_URL: The base URL of your LiteLLM proxy

Usage:
    export OPENAI_API_KEY="sk-1234"
    export OPENAI_BASE_URL="http://localhost:4000"
    uv run load_test.py

    # Run only quick tests
    uv run load_test.py --quick

    # Enable debug logging (shows successful requests)
    uv run load_test.py --debug

    # Get help
    uv run load_test.py --help
"""

import asyncio
import json
import time
import logging
import statistics
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse
import sys
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("results/load_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Configure OpenAI client logging to show 429 errors but suppress 200 responses
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.INFO)  # Allow INFO level to see 429 responses


# Add a custom filter to suppress 200 responses but allow 429 responses
class OpenAIResponseFilter(logging.Filter):
    def filter(self, record):
        # Suppress 200 responses but allow 429 and other error responses
        if "HTTP/1.1 200 OK" in record.getMessage():
            return False
        return True


# Apply the filter to OpenAI logger
openai_logger.addFilter(OpenAIResponseFilter())

# Configure HTTP client logging to show 429 errors but suppress 200 responses
http_logger = logging.getLogger("httpx")
http_logger.setLevel(logging.INFO)  # Allow INFO level to see 429 responses

# Suppress urllib3 logging (used by some HTTP clients)
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.WARNING)

# Suppress any other HTTP-related loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)


# Add a custom filter to suppress 200 responses but allow 429 responses
class HTTPResponseFilter(logging.Filter):
    def filter(self, record):
        # Suppress 200 responses but allow 429 and other error responses
        if "HTTP/1.1 200 OK" in record.getMessage():
            return False
        return True


# Apply the filter to httpx logger
http_logger.addFilter(HTTPResponseFilter())


@dataclass
class TestResult:
    """Container for test results"""

    test_name: str
    success: bool
    duration: float
    response_length: int
    error: Optional[str] = None
    model: Optional[str] = None
    timestamp: Optional[datetime] = None
    is_rate_limited: bool = False


class LoadTester:
    """Main load testing class"""

    def __init__(self):
        # Get configuration from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY", "sk-1234")
        self.base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:4000")
        self.client: Optional[AsyncOpenAI] = None
        self.results: List[TestResult] = []
        self.models: List[str] = []

        # LiteLLM configuration limits (from litellm.yaml)
        self.global_max_parallel_requests = 20
        self.request_timeout = 600  # 10 minutes
        self.database_connection_pool_limit = 30

        # Test prompts
        self.quick_prompts = [
            "Hello, how are you?",
            "What is 2+2?",
            "Say 'test'",
            "Count to 3",
            "What color is the sky?",
        ]

        self.long_prompts = [
            "Write a detailed 500-word essay about the history of artificial intelligence, including key milestones, major contributors, and future implications.",
            "Explain the concept of machine learning in great detail, covering supervised learning, unsupervised learning, reinforcement learning, and provide specific examples for each.",
            "Describe the process of training a neural network from scratch, including data preparation, model architecture design, loss functions, optimization algorithms, and evaluation metrics.",
            "Write a comprehensive analysis of the pros and cons of different programming languages for AI development, including Python, R, Julia, and C++.",
            "Create a detailed technical specification for building a chatbot system, including architecture, data flow, API design, security considerations, and scalability requirements.",
        ]

    async def __aenter__(self):
        """Async context manager entry"""
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout,  # Use LiteLLM's request timeout
            max_retries=0,  # Disable automatic retries to get accurate timing
        )
        # Load available models dynamically
        await self.load_models()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.close()

    async def load_models(self):
        """Load available models from the /models endpoint"""
        logger.info("Fetching available models...")
        models_response = await self.client.models.list()
        all_models = [model.id for model in models_response.data]

        if not all_models:
            raise RuntimeError("No models found in the LiteLLM service")

        # Separate completion and embedding models
        self.models = []
        self.embedding_models = []

        # Known embedding models from litellm.yaml
        known_embedding_models = [
            "nomic-embed-text-v1-GGUF",
            "nomic-embed-text-v2-moe-GGUF",
        ]

        for model in all_models:
            if model in known_embedding_models:
                self.embedding_models.append(model)
            else:
                self.models.append(model)

        logger.info(f"Found {len(self.models)} completion models: {self.models}")
        if self.embedding_models:
            logger.info(
                f"Found {len(self.embedding_models)} embedding models: {self.embedding_models}"
            )
        logger.info(
            f"LiteLLM limits: max_parallel={self.global_max_parallel_requests}, timeout={self.request_timeout}s, db_pool={self.database_connection_pool_limit}"
        )

    def get_largest_model(self) -> str:
        """Get the largest model (prefer gpt-oss-120b-mxfp-GGUF if available)"""
        # Prefer the largest model if available
        preferred_models = [
            "gpt-oss-120b-mxfp-GGUF",
            "Qwen3-30B-A3B-Instruct-2507-GGUF",
            "Qwen3-Coder-30B-A3B-Instruct-GGUF",
        ]

        for model in preferred_models:
            if model in self.models:
                return model

        # Return the first available model if none of the preferred ones are found
        return self.models[0]

    def get_smaller_models(self) -> List[str]:
        """Get smaller models for quick response tests"""
        smaller_models = [
            "Gemma-3-4b-it-GGUF",
            "DeepSeek-Qwen3-8B-GGUF",
        ]

        available_smaller = [model for model in smaller_models if model in self.models]
        if not available_smaller:
            # If no smaller models available, use the last model in the list
            return [self.models[-1]]

        return available_smaller

    async def make_embedding_request(self, model: str, text: str) -> TestResult:
        """Make a single embedding API request"""
        start_time = time.time()
        timestamp = datetime.now()

        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text,
            )

            duration = time.time() - start_time
            embedding = response.data[0].embedding if response.data else []

            # Log successful requests at debug level to reduce noise
            logger.debug(f"Embedding request to {model} succeeded in {duration:.2f}s")

            return TestResult(
                test_name="embedding_request",
                success=True,
                duration=duration,
                response_length=len(embedding),
                model=model,
                timestamp=timestamp,
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.warning(
                f"Embedding request to {model} timed out after {duration:.2f}s"
            )
            return TestResult(
                test_name="embedding_request",
                success=False,
                duration=duration,
                response_length=0,
                error="Request timeout",
                model=model,
                timestamp=timestamp,
            )
        except Exception as e:
            duration = time.time() - start_time
            error_str = str(e)
            is_rate_limited = False

            # Log different error types with appropriate levels
            if "429" in error_str or "rate limit" in error_str.lower():
                logger.warning(f"Rate limit hit for embedding {model}: {error_str}")
                is_rate_limited = True
            elif "500" in error_str or "internal server error" in error_str.lower():
                logger.error(f"Server error for embedding {model}: {error_str}")
            elif "connection" in error_str.lower() or "refused" in error_str.lower():
                logger.error(f"Connection error for embedding {model}: {error_str}")
            else:
                logger.warning(f"Embedding request to {model} failed: {error_str}")

            return TestResult(
                test_name="embedding_request",
                success=False,
                duration=duration,
                response_length=0,
                error=error_str,
                model=model,
                timestamp=timestamp,
                is_rate_limited=is_rate_limited,
            )

    async def make_request(
        self, model: str, prompt: str, max_tokens: int = 100
    ) -> TestResult:
        """Make a single API request"""
        start_time = time.time()
        timestamp = datetime.now()

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )

            duration = time.time() - start_time
            content = response.choices[0].message.content or ""

            # Log successful requests at debug level to reduce noise
            logger.debug(f"Request to {model} succeeded in {duration:.2f}s")

            return TestResult(
                test_name="single_request",
                success=True,
                duration=duration,
                response_length=len(content),
                model=model,
                timestamp=timestamp,
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.warning(f"Request to {model} timed out after {duration:.2f}s")
            return TestResult(
                test_name="single_request",
                success=False,
                duration=duration,
                response_length=0,
                error="Request timeout",
                model=model,
                timestamp=timestamp,
            )
        except Exception as e:
            duration = time.time() - start_time
            error_str = str(e)
            is_rate_limited = False

            # Log different error types with appropriate levels
            if "429" in error_str or "rate limit" in error_str.lower():
                logger.warning(f"Rate limit hit for {model}: {error_str}")
                is_rate_limited = True
            elif "500" in error_str or "internal server error" in error_str.lower():
                logger.error(f"Server error for {model}: {error_str}")
            elif "connection" in error_str.lower() or "refused" in error_str.lower():
                logger.error(f"Connection error for {model}: {error_str}")
            else:
                logger.warning(f"Request to {model} failed: {error_str}")

            return TestResult(
                test_name="single_request",
                success=False,
                duration=duration,
                response_length=0,
                error=error_str,
                model=model,
                timestamp=timestamp,
                is_rate_limited=is_rate_limited,
            )

    async def test_parallel_large_model(
        self, num_requests: int = 12
    ) -> List[TestResult]:
        """Test multiple parallel calls to the largest model"""
        largest_model = self.get_largest_model()
        logger.info(f"Testing {num_requests} parallel calls to {largest_model}")

        tasks = []
        for i in range(num_requests):
            prompt = (
                f"Test request {i+1}: "
                f"{self.quick_prompts[i % len(self.quick_prompts)]}"
            )
            task = self.make_request(largest_model, prompt)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    TestResult(
                        test_name="parallel_large_model",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model=largest_model,
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "parallel_large_model"
                processed_results.append(result)

        return processed_results

    async def test_parallel_multi_model(
        self, requests_per_model: int = 3
    ) -> List[TestResult]:
        """Test parallel calls to multiple models simultaneously"""
        logger.info(
            f"Testing {requests_per_model} requests per model across {len(self.models)} models"
        )

        tasks = []
        for model in self.models:
            for i in range(requests_per_model):
                prompt = f"Multi-model test {i+1} for {model}: {self.quick_prompts[i % len(self.quick_prompts)]}"
                task = self.make_request(model, prompt)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                model_idx = i // requests_per_model
                processed_results.append(
                    TestResult(
                        test_name="parallel_multi_model",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model=(
                            self.models[model_idx]
                            if model_idx < len(self.models)
                            else "unknown"
                        ),
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "parallel_multi_model"
                processed_results.append(result)

        return processed_results

    async def test_quick_responses(self, num_requests: int = 16) -> List[TestResult]:
        """Test calls that should elicit quick responses"""
        smaller_models = self.get_smaller_models()
        model = smaller_models[0]  # Use the first available smaller model
        logger.info(f"Testing {num_requests} quick response requests using {model}")

        tasks = []
        for i in range(num_requests):
            prompt = self.quick_prompts[i % len(self.quick_prompts)]
            task = self.make_request(model, prompt, max_tokens=50)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    TestResult(
                        test_name="quick_responses",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model=model,
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "quick_responses"
                processed_results.append(result)

        return processed_results

    async def test_long_responses(self, num_requests: int = 5) -> List[TestResult]:
        """Test calls that should elicit long responses"""
        largest_model = self.get_largest_model()
        logger.info(
            f"Testing {num_requests} long response requests using {largest_model}"
        )

        tasks = []
        for i in range(num_requests):
            prompt = self.long_prompts[i % len(self.long_prompts)]
            task = self.make_request(largest_model, prompt, max_tokens=500)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    TestResult(
                        test_name="long_responses",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model=largest_model,
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "long_responses"
                processed_results.append(result)

        return processed_results

    async def test_memory_persistence(
        self, model: Optional[str] = None
    ) -> List[TestResult]:
        """Test that recent requests to the same model are still in memory"""
        if model is None:
            model = self.get_largest_model()
        logger.info(f"Testing memory persistence for model {model}")

        # First, make a request to load the model into memory
        logger.info("Making initial request to load model into memory...")
        initial_result = await self.make_request(
            model, "This is the initial request to load the model."
        )

        # Wait a bit to simulate some time passing
        await asyncio.sleep(2)

        # Then make several quick requests to test if model is still loaded
        test_prompts = [
            "What was the previous request about?",
            "Are you still loaded in memory?",
            "Can you remember our conversation?",
            "What's 5+5?",
            "Quick test of memory",
        ]

        tasks = []
        for prompt in test_prompts:
            task = self.make_request(model, prompt, max_tokens=100)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = [initial_result]
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    TestResult(
                        test_name="memory_persistence",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model=model,
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "memory_persistence"
                processed_results.append(result)

        return processed_results

    async def test_parallel_limit_boundary(self) -> List[TestResult]:
        """Test boundary conditions around the global_max_parallel_requests limit"""
        logger.info("Testing parallel request limit boundary conditions...")

        # Test 1: Exactly at the limit (should work fine)
        logger.info(
            f"Testing exactly {self.global_max_parallel_requests} parallel requests..."
        )
        at_limit_tasks = []
        largest_model = self.get_largest_model()
        for i in range(self.global_max_parallel_requests):
            prompt = f"At limit test {i+1}: {self.quick_prompts[i % len(self.quick_prompts)]}"
            task = self.make_request(largest_model, prompt, max_tokens=50)
            at_limit_tasks.append(task)

        at_limit_results = await asyncio.gather(*at_limit_tasks, return_exceptions=True)

        # Test 2: Just over the limit (should trigger queuing/rate limiting)
        logger.info(
            f"Testing {self.global_max_parallel_requests + 1} parallel requests (just over limit)..."
        )
        over_limit_tasks = []
        for i in range(self.global_max_parallel_requests + 1):
            prompt = f"Over limit test {i+1}: {self.quick_prompts[i % len(self.quick_prompts)]}"
            task = self.make_request(largest_model, prompt, max_tokens=50)
            over_limit_tasks.append(task)

        over_limit_results = await asyncio.gather(
            *over_limit_tasks, return_exceptions=True
        )

        # Test 3: Test with multiple models hitting the limit simultaneously
        logger.info("Testing multiple models hitting parallel limit simultaneously...")
        multi_model_tasks = []
        smaller_models = self.get_smaller_models()
        if len(smaller_models) > 0:
            # Use 3 requests per model to stay under total limit but test per-model limits
            for model in smaller_models[:2]:  # Test with first 2 smaller models
                for i in range(3):
                    prompt = f"Multi-model test {i+1} for {model}: {self.quick_prompts[i % len(self.quick_prompts)]}"
                    task = self.make_request(model, prompt, max_tokens=50)
                    multi_model_tasks.append(task)

        multi_model_results = await asyncio.gather(
            *multi_model_tasks, return_exceptions=True
        )

        # Process all results
        all_results = []

        # Process at-limit results
        for i, result in enumerate(at_limit_results):
            if isinstance(result, Exception):
                all_results.append(
                    TestResult(
                        test_name="parallel_limit_at_boundary",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model=largest_model,
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "parallel_limit_at_boundary"
                all_results.append(result)

        # Process over-limit results
        for i, result in enumerate(over_limit_results):
            if isinstance(result, Exception):
                all_results.append(
                    TestResult(
                        test_name="parallel_limit_over_boundary",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model=largest_model,
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "parallel_limit_over_boundary"
                all_results.append(result)

        # Process multi-model results
        for i, result in enumerate(multi_model_results):
            if isinstance(result, Exception):
                model = (
                    smaller_models[i // 3]
                    if i // 3 < len(smaller_models)
                    else "unknown"
                )
                all_results.append(
                    TestResult(
                        test_name="parallel_limit_multi_model",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model=model,
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "parallel_limit_multi_model"
                all_results.append(result)

        return all_results

    async def test_embedding_models(self) -> List[TestResult]:
        """Test embedding models with appropriate API calls"""
        if not self.embedding_models:
            logger.info("No embedding models found, skipping embedding tests")
            return []

        logger.info(f"Testing embedding models: {self.embedding_models}")

        # Test texts for embedding
        test_texts = [
            "This is a test sentence for embedding.",
            "Another test with different content.",
            "Short text.",
            "This is a much longer text that contains more words and should generate a different embedding vector compared to the shorter texts above.",
            "Technical documentation about machine learning and artificial intelligence systems.",
        ]

        all_results = []

        for model in self.embedding_models:
            logger.info(f"Testing embedding model: {model}")

            # Test individual embedding requests
            for i, text in enumerate(test_texts):
                result = await self.make_embedding_request(model, text)
                result.test_name = "embedding_individual"
                all_results.append(result)

            # Test parallel embedding requests
            logger.info(f"Testing parallel embedding requests for {model}")
            parallel_tasks = []
            for i in range(
                6
            ):  # Test with 6 parallel requests (under embedding limit of 8)
                text = (
                    f"Parallel embedding test {i+1}: {test_texts[i % len(test_texts)]}"
                )
                task = self.make_embedding_request(model, text)
                parallel_tasks.append(task)

            parallel_results = await asyncio.gather(
                *parallel_tasks, return_exceptions=True
            )

            # Process parallel results
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    all_results.append(
                        TestResult(
                            test_name="embedding_parallel",
                            success=False,
                            duration=0,
                            response_length=0,
                            error=str(result),
                            model=model,
                            timestamp=datetime.now(),
                        )
                    )
                else:
                    result.test_name = "embedding_parallel"
                    all_results.append(result)

        return all_results

    async def test_stress_conditions(self) -> List[TestResult]:
        """Test various stress conditions that might cause lockups"""
        logger.info("Testing stress conditions...")

        # Test 1: Rapid fire requests (exceeding global_max_parallel_requests)
        largest_model = self.get_largest_model()
        logger.info(
            f"Testing rapid fire requests (exceeding parallel limit of {self.global_max_parallel_requests})..."
        )
        rapid_tasks = []
        # Test with 2x the parallel limit to trigger queuing/rate limiting
        num_rapid_requests = self.global_max_parallel_requests * 2
        for i in range(num_rapid_requests):
            prompt = f"Rapid fire test {i+1}: {self.quick_prompts[i % len(self.quick_prompts)]}"
            task = self.make_request(largest_model, prompt, max_tokens=50)
            rapid_tasks.append(task)

        rapid_results = await asyncio.gather(*rapid_tasks, return_exceptions=True)

        # Test 2: Mixed request types simultaneously
        logger.info("Testing mixed request types...")
        mixed_tasks = []

        # Add some quick requests
        smaller_models = self.get_smaller_models()
        for i in range(3):
            mixed_tasks.append(
                self.make_request(
                    smaller_models[0], self.quick_prompts[i], max_tokens=50
                )
            )

        # Add some long requests
        for i in range(2):
            mixed_tasks.append(
                self.make_request(largest_model, self.long_prompts[i], max_tokens=200)
            )

        # Add some medium requests
        if len(self.models) > 1:
            medium_model = self.models[1] if len(self.models) > 1 else largest_model
            for i in range(3):
                mixed_tasks.append(
                    self.make_request(
                        medium_model,
                        f"Medium test {i+1}",
                        max_tokens=100,
                    )
                )

        mixed_results = await asyncio.gather(*mixed_tasks, return_exceptions=True)

        # Process all results
        all_results = []

        # Process rapid fire results
        for i, result in enumerate(rapid_results):
            if isinstance(result, Exception):
                all_results.append(
                    TestResult(
                        test_name="stress_rapid_fire",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model=largest_model,
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "stress_rapid_fire"
                all_results.append(result)

        # Process mixed results
        for i, result in enumerate(mixed_results):
            if isinstance(result, Exception):
                all_results.append(
                    TestResult(
                        test_name="stress_mixed",
                        success=False,
                        duration=0,
                        response_length=0,
                        error=str(result),
                        model="mixed",
                        timestamp=datetime.now(),
                    )
                )
            else:
                result.test_name = "stress_mixed"
                all_results.append(result)

        return all_results

    def analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """Analyze test results and generate statistics"""
        if not results:
            return {"error": "No results to analyze"}

        # Group by test name
        by_test = {}
        for result in results:
            if result.test_name not in by_test:
                by_test[result.test_name] = []
            by_test[result.test_name].append(result)

        analysis = {}

        for test_name, test_results in by_test.items():
            successful = [r for r in test_results if r.success]
            failed = [r for r in test_results if not r.success]
            rate_limited = [r for r in test_results if r.is_rate_limited]

            durations = [r.duration for r in successful if r.duration > 0]

            # Analyze rate limiting conditions
            rate_limit_conditions = {}
            for result in rate_limited:
                condition_key = f"{result.model}_{result.test_name}"
                if condition_key not in rate_limit_conditions:
                    rate_limit_conditions[condition_key] = {
                        "model": result.model,
                        "test": result.test_name,
                        "count": 0,
                        "avg_duration": 0,
                        "durations": [],
                    }
                rate_limit_conditions[condition_key]["count"] += 1
                rate_limit_conditions[condition_key]["durations"].append(
                    result.duration
                )

            # Calculate average duration for each rate limit condition
            for condition in rate_limit_conditions.values():
                condition["avg_duration"] = (
                    statistics.mean(condition["durations"])
                    if condition["durations"]
                    else 0
                )

            analysis[test_name] = {
                "total_requests": len(test_results),
                "successful": len(successful),
                "failed": len(failed),
                "rate_limited": len(rate_limited),
                "rate_limit_conditions": rate_limit_conditions,
                "success_rate": (
                    len(successful) / len(test_results) * 100 if test_results else 0
                ),
                "avg_duration": statistics.mean(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "median_duration": statistics.median(durations) if durations else 0,
                "errors": [r.error for r in failed if r.error],
                "timeouts": len(
                    [r for r in failed if "timeout" in (r.error or "").lower()]
                ),
                "lockups": len(
                    [
                        r
                        for r in failed
                        if any(
                            keyword in (r.error or "").lower()
                            for keyword in ["connection", "refused", "reset", "aborted"]
                        )
                    ]
                ),
            }

        return analysis

    def print_analysis(self, analysis: Dict[str, Any]):
        """Print analysis results in a readable format"""
        print("\n" + "=" * 80)
        print("LOAD TEST ANALYSIS")
        print("=" * 80)

        for test_name, stats in analysis.items():
            print(f"\n{test_name.upper()}:")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Successful: {stats['successful']}")
            print(f"  Failed: {stats['failed']}")
            if stats["rate_limited"] > 0:
                print(f"  Rate Limited: {stats['rate_limited']}")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")

            if stats["avg_duration"] > 0:
                print(f"  Average Duration: {stats['avg_duration']:.2f}s")
                print(f"  Min Duration: {stats['min_duration']:.2f}s")
                print(f"  Max Duration: {stats['max_duration']:.2f}s")
                print(f"  Median Duration: {stats['median_duration']:.2f}s")

            if stats["timeouts"] > 0:
                print(f"  Timeouts: {stats['timeouts']}")

            if stats["rate_limited"] > 0:
                print(
                    f"  Rate Limited: {stats['rate_limited']} (429 responses - no retries)"
                )
                # Show rate limiting conditions
                for condition in stats["rate_limit_conditions"].values():
                    print(
                        f"    - {condition['model']} in {condition['test']}: "
                        f"{condition['count']} hits (avg {condition['avg_duration']:.2f}s)"
                    )

            if stats["lockups"] > 0:
                print(f"  Potential Lockups: {stats['lockups']}")

            if stats["errors"]:
                unique_errors = list(set(stats["errors"]))
                print(f"  Unique Errors: {len(unique_errors)}")
                for error in unique_errors[:5]:  # Show first 5 unique errors
                    print(f"    - {error}")

        # Print overall rate limiting summary
        self.print_rate_limit_summary(analysis)

    def print_rate_limit_summary(self, analysis: Dict[str, Any]):
        """Print a summary of rate limiting patterns across all tests"""
        all_rate_limits = []
        for test_name, stats in analysis.items():
            if stats["rate_limited"] > 0:
                for condition in stats["rate_limit_conditions"].values():
                    all_rate_limits.append(
                        {
                            "test": test_name,
                            "model": condition["model"],
                            "count": condition["count"],
                            "avg_duration": condition["avg_duration"],
                        }
                    )

        if all_rate_limits:
            print("\n" + "=" * 80)
            print("RATE LIMITING SUMMARY")
            print("=" * 80)

            # Group by model to see which models are most affected
            model_limits = {}
            for limit in all_rate_limits:
                model = limit["model"]
                if model not in model_limits:
                    model_limits[model] = {"total_hits": 0, "tests": []}
                model_limits[model]["total_hits"] += limit["count"]
                model_limits[model]["tests"].append(limit)

            print("Models most affected by rate limiting:")
            for model, data in sorted(
                model_limits.items(), key=lambda x: x[1]["total_hits"], reverse=True
            ):
                print(f"  {model}: {data['total_hits']} total rate limit hits")
                for test in data["tests"]:
                    print(
                        f"    - {test['test']}: {test['count']} hits (avg {test['avg_duration']:.2f}s)"
                    )

            print(
                f"\nTotal rate limit hits across all tests: {sum(limit['count'] for limit in all_rate_limits)}"
            )

    async def run_all_tests(self):
        """Run all load tests"""
        logger.info("Starting comprehensive load testing...")

        all_results = []

        # Test 1: Parallel calls to largest model
        logger.info("\n=== Test 1: Parallel Large Model Calls ===")
        results1 = await self.test_parallel_large_model(12)
        all_results.extend(results1)

        # Test 2: Parallel calls to multiple models
        logger.info("\n=== Test 2: Parallel Multi-Model Calls ===")
        results2 = await self.test_parallel_multi_model(3)
        all_results.extend(results2)

        # Test 3: Quick responses
        logger.info("\n=== Test 3: Quick Response Calls ===")
        results3 = await self.test_quick_responses(16)
        all_results.extend(results3)

        # Test 4: Long responses
        logger.info("\n=== Test 4: Long Response Calls ===")
        results4 = await self.test_long_responses(5)
        all_results.extend(results4)

        # Test 5: Memory persistence
        logger.info("\n=== Test 5: Memory Persistence ===")
        results5 = await self.test_memory_persistence()
        all_results.extend(results5)

        # Test 6: Parallel limit boundary testing
        logger.info("\n=== Test 6: Parallel Limit Boundary Testing ===")
        results6 = await self.test_parallel_limit_boundary()
        all_results.extend(results6)

        # Test 7: Embedding models
        logger.info("\n=== Test 7: Embedding Models ===")
        results7 = await self.test_embedding_models()
        all_results.extend(results7)

        # Test 8: Stress conditions
        logger.info("\n=== Test 8: Stress Conditions ===")
        results8 = await self.test_stress_conditions()
        all_results.extend(results8)

        # Store all results
        self.results = all_results

        # Analyze and print results
        analysis = self.analyze_results(all_results)
        self.print_analysis(analysis)

        # Save detailed results to file
        with open("results/load_test_results.json", "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis,
                    "raw_results": [
                        {
                            "test_name": r.test_name,
                            "success": r.success,
                            "duration": r.duration,
                            "response_length": r.response_length,
                            "error": r.error,
                            "model": r.model,
                            "timestamp": (
                                r.timestamp.isoformat() if r.timestamp else None
                            ),
                        }
                        for r in all_results
                    ],
                },
                f,
                indent=2,
            )

        logger.info(
            "\nLoad testing completed. Results saved to results/load_test_results.json"
        )
        logger.info(f"Total requests made: {len(all_results)}")

        return analysis


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Load test LiteLLM models")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows successful requests)",
    )

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled - will show successful requests")

    # Get configuration from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    if not base_url:
        print("Error: OPENAI_BASE_URL environment variable is required")
        sys.exit(1)

    print(f"Starting load test against: {base_url}")
    print(f"Using API key: {api_key[:8]}...")

    try:
        async with LoadTester() as tester:
            if args.quick:
                # Run only quick tests
                logger.info("Running quick tests only...")
                results = []
                results.extend(await tester.test_quick_responses(10))
                results.extend(await tester.test_parallel_large_model(8))
                # Include embedding tests in quick mode if embedding models exist
                results.extend(await tester.test_embedding_models())

                analysis = tester.analyze_results(results)
                tester.print_analysis(analysis)
            else:
                # Run all tests
                await tester.run_all_tests()
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        print(f"Error: {e}")
        print("Make sure your LiteLLM service is running and accessible.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
