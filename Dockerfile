FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN addgroup --system alpha && adduser --system --ingroup alpha alpha
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN mkdir -p /app/data /app/artifacts && chown -R alpha:alpha /app
USER alpha
EXPOSE 8000
CMD ["uvicorn", "alpha_engine.api:app", "--host", "0.0.0.0", "--port", "8000"]

