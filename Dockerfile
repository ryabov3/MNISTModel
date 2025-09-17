FROM python:3.12.6

WORKDIR /app

COPY pyproject.toml /app
COPY uv.lock /app

RUN pip install uv
RUN uv sync --frozen

COPY . /app

CMD ["python", "/app/main.py"]
