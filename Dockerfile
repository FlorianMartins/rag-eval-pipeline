# Reproducible evaluation image: the same bits run on a laptop, in CI and in a scheduled job.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
RUN pip install . && useradd --create-home --uid 10001 eval

COPY knowledge_base ./knowledge_base
COPY evals ./evals
RUN mkdir reports && chown eval:eval reports

USER eval
ENTRYPOINT ["rageval"]
CMD ["run", "--baseline", "evals/baseline.json", "--out", "reports"]
