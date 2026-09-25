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
# The embedding model is baked in at the revision pinned in app/config.toml, so the
# container evaluates offline and every run uses byte-identical weights.
ENV HF_HOME=/opt/hf
RUN pip install '.[embeddings]' \
 && python -c "import tomllib; from huggingface_hub import snapshot_download; \
h = tomllib.load(open('app/config.toml', 'rb'))['retrieval']['hybrid']; \
snapshot_download(h['embedding_model'], revision=h['embedding_revision'])" \
 && chmod -R a+rX /opt/hf \
 && useradd --create-home --uid 10001 eval
ENV HF_HUB_OFFLINE=1

COPY knowledge_base ./knowledge_base
COPY evals ./evals
RUN mkdir reports && chown eval:eval reports

USER eval
ENTRYPOINT ["rageval"]
CMD ["run", "--baseline", "evals/baseline.json", "--out", "reports"]
