# Patches for Submodules

This directory contains patches that are applied to git submodules during Docker builds.

## pgvector-768-dimensions.patch

This patch modifies the `litellm-pgvector` submodule to support 768-dimensional embeddings instead of the default 1536 dimensions.

**Applied to:** `litellm-pgvector/prisma/schema.prisma`

**Why needed:** The nomic-embed-text-v2-moe model generates 768-dimensional embeddings, but the upstream litellm-pgvector defaults to 1536 dimensions (for OpenAI's text-embedding-ada-002).

**Persistence:** This patch is applied during the Docker build process (see `Dockerfile.pgvector`), so it persists across submodule updates.

## pgvector-fix-file-counts-sql.patch

This patch fixes a SQL bug in the `litellm-pgvector` submodule where UPDATE statements attempted to assign to the `file_counts` column twice, causing a PostgreSQL error: "multiple assignments to same column".

**Applied to:** `litellm-pgvector/main.py`

**Why needed:** The upstream code has a bug in two UPDATE queries (lines ~371-380 and ~471-480) that try to set `file_counts` twice in the same SET clause. This patch fixes it by nesting the `jsonb_set` calls properly.

**Persistence:** This patch is applied during the Docker build process (see `Dockerfile.pgvector`), so it persists across submodule updates.

## Updating Patches After Submodule Updates

If you update the `litellm-pgvector` submodule and the patch fails to apply:

1. Manually make the required changes to the submodule files
2. Regenerate the patch:
   ```bash
   cd litellm-pgvector
   git diff > ../patches/pgvector-768-dimensions.patch
   ```
3. Revert your manual changes:
   ```bash
   git checkout .
   ```
4. Rebuild the Docker image

