.PHONY: install lint test demo api

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests
	mypy src

test:
	pytest --cov=alpha_engine --cov-report=term-missing

demo:
	alpha demo --days 1200

api:
	uvicorn alpha_engine.api:app --host 0.0.0.0 --port 8000

