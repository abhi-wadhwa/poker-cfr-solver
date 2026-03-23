.PHONY: run train test lint install clean benchmark

# Install dependencies
install:
	pip install -e ".[dev]"

# Run Streamlit interactive dashboard
run:
	streamlit run src/viz/app.py

# Train CFR on Kuhn Poker (default 10,000 iterations)
train:
	python -m src.cli train --game kuhn --algo cfr --iterations 10000

# Train CFR on Leduc Hold'em
train-leduc:
	python -m src.cli train --game leduc --algo cfr --iterations 5000

# Show Nash equilibrium strategy
show:
	python -m src.cli show --game kuhn --algo cfr --iterations 10000

# Run benchmark comparing all algorithms
benchmark:
	python -m src.cli benchmark --game kuhn --iterations 10000

# Run tests with coverage
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Lint with ruff
lint:
	ruff check src/ tests/

# Format with ruff
format:
	ruff format src/ tests/

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
