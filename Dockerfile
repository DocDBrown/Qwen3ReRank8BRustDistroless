# syntax=docker/dockerfile:1.7

############################
# syslibs: runtime deps
############################
FROM debian:bookworm-slim AS syslibs
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

############################
# ort: ONNX Runtime prebuilt
############################
FROM debian:bookworm-slim AS ort
WORKDIR /opt/onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl tar \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o onnxruntime.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-aarch64-1.22.0.tgz" \
    && tar -xzf onnxruntime.tgz --strip-components=1 \
    && rm -f onnxruntime.tgz

############################
# model: use local onnx files
############################
FROM scratch AS model
COPY onnx /download/onnx

############################
# builder: compile rust binary
############################
FROM rust:1-bookworm AS builder
WORKDIR /src

# IMPORTANT: Ensure this matches [package] name in Cargo.toml
ARG BIN_NAME=qwen-rerank

# Cache deps first
COPY Cargo.toml ./

# Create dummy main to cache dependencies
RUN mkdir -p src && echo "fn main(){}" > src/main.rs \
    && cargo build --release || true

# Now copy the real source and build
COPY src ./src

# Force touch main.rs to ensure rebuild, create /out, and copy binary
RUN touch src/main.rs && cargo build --release \
    && mkdir -p /out \
    && (cp "target/release/${BIN_NAME}" /out/server || cp "target/release/$(grep -oE '^name = \"[^\"]+' Cargo.toml | cut -d'\"' -f2)" /out/server)

############################
# runtime: distroless
############################
FROM gcr.io/distroless/cc-debian12:nonroot
WORKDIR /app

# Copy binary
COPY --from=builder /out/server /app/server

# ONNX Runtime shared libs
COPY --from=ort /opt/onnxruntime/lib /opt/onnxruntime/lib

# libgomp for ORT/OpenMP usage
COPY --from=syslibs /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libgomp.so.1

# Model + tokenizer
COPY --from=model /download/onnx /app/onnx

# Set library path so binary finds libonnxruntime.so
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib
ENV RERANKER_MODEL_PATH=/app/onnx/model.onnx
ENV RERANKER_TOKENIZER_PATH=/app/onnx/tokenizer.json

EXPOSE 8982

ENTRYPOINT ["/app/server"]