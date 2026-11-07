# The benchmark harness only: client, replayer, proxy and analysis. It doesn't include vLLM or a GPU
# runtime, so point it at a vLLM server that's already running.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.12.3

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["anton"]
CMD ["--help"]
