FROM postgres:17-trixie

# Update package lists and install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    postgresql-server-dev-17 \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector following the official README instructions
RUN cd /tmp \
    && git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && cd .. \
    && rm -rf pgvector

# Copy initialization script to enable vector extension
COPY init.sql /docker-entrypoint-initdb.d/
