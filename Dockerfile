FROM python:3.12.6 AS model-training
WORKDIR /app
COPY pyproject.toml uv.lock /app
RUN pip install uv && uv sync
ENV PATH="/app/.venv/bin:$PATH"
COPY . /app
CMD ["python", "/app/train_model.py"]
