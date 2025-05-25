.PHONY: help clean install install-dev test lint format build check-build upload-test upload-prod release check-name
.DEFAULT_GOAL := help

# Variables
PACKAGE_NAME := judgy
PYTHON := python3
PIP := pip3

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build artifacts and cache files
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Clean complete"

install: ## Install package dependencies
	@echo "📦 Installing dependencies..."
	$(PIP) install -e .
	@echo "✅ Installation complete"

install-dev: ## Install package with development dependencies
	@echo "📦 Installing development dependencies..."
	$(PIP) install -e .[dev,plotting]
	@echo "✅ Development installation complete"

install-build-tools: ## Install build and upload tools
	@echo "🔧 Installing build tools..."
	$(PIP) install build twine
	@echo "✅ Build tools installed"

test: ## Run tests
	@echo "🧪 Running tests..."
	pytest tests/ -v
	@echo "✅ Tests complete"

test-cov: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	pytest tests/ --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term
	@echo "✅ Tests with coverage complete"

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	flake8 src/$(PACKAGE_NAME)/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/$(PACKAGE_NAME)/
	@echo "✅ Linting complete"

format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	black src/$(PACKAGE_NAME)/ tests/
	isort src/$(PACKAGE_NAME)/ tests/
	@echo "✅ Formatting complete"

check-author: ## Check if author information is updated
	@echo "👤 Checking author information..."
	@if grep -q "Your Name" pyproject.toml; then \
		echo "❌ Please update author information in pyproject.toml"; \
		echo "   Replace 'Your Name' and 'your.email@example.com' with your actual details"; \
		exit 1; \
	else \
		echo "✅ Author information looks good"; \
	fi

check-name: ## Check if package name is available on PyPI
	@echo "🔍 Checking if package name '$(PACKAGE_NAME)' is available on PyPI..."
	@if curl -s https://pypi.org/project/$(PACKAGE_NAME)/ | grep -q "Not Found"; then \
		echo "✅ Package name '$(PACKAGE_NAME)' appears to be available"; \
	else \
		echo "⚠️  Package name '$(PACKAGE_NAME)' might already exist on PyPI"; \
		echo "   Check: https://pypi.org/project/$(PACKAGE_NAME)/"; \
	fi

build: clean install-build-tools check-author ## Build the package
	@echo "🏗️  Building package..."
	$(PYTHON) -m build
	@echo "✅ Build complete"
	@echo "📦 Generated files:"
	@ls -la dist/

check-build: build ## Build and check the package
	@echo "🔍 Checking built package..."
	$(PYTHON) -m twine check dist/*
	@echo "✅ Package check complete"

upload-test: check-build ## Upload to TestPyPI
	@echo "🚀 Uploading to TestPyPI..."
	@echo "⚠️  You'll need to enter your TestPyPI credentials"
	@echo "   Username: __token__"
	@echo "   Password: your-testpypi-api-token"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "✅ Upload to TestPyPI complete"
	@echo "🔗 Check your package at: https://test.pypi.org/project/$(PACKAGE_NAME)/"
	@echo "📥 Test install with: pip install --index-url https://test.pypi.org/simple/ $(PACKAGE_NAME)"

upload-prod: check-build ## Upload to PyPI (production)
	@echo "🚀 Uploading to PyPI..."
	@echo "⚠️  You'll need to enter your PyPI credentials"
	@echo "   Username: __token__"
	@echo "   Password: your-pypi-api-token"
	@read -p "Are you sure you want to upload to production PyPI? (y/N): " confirm && [ "$$confirm" = "y" ]
	$(PYTHON) -m twine upload dist/*
	@echo "✅ Upload to PyPI complete"
	@echo "🔗 Check your package at: https://pypi.org/project/$(PACKAGE_NAME)/"
	@echo "📥 Install with: pip install $(PACKAGE_NAME)"

pre-release: ## Run all pre-release checks
	@echo "🔍 Running pre-release checks..."
	$(MAKE) clean
	$(MAKE) install-dev
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-cov
	$(MAKE) check-author
	$(MAKE) check-name
	@echo "✅ All pre-release checks passed"

release-test: pre-release upload-test ## Full release workflow to TestPyPI
	@echo "🎉 Test release complete!"

release-prod: pre-release upload-prod ## Full release workflow to PyPI
	@echo "🎉 Production release complete!"

# Quick commands
quick-test: clean build upload-test ## Quick test release (skip some checks)

quick-prod: clean build upload-prod ## Quick production release (skip some checks)

# Development helpers
dev-setup: install-dev install-build-tools ## Set up development environment
	@echo "🛠️  Development environment ready!"

version-bump-patch: ## Bump patch version (0.1.0 -> 0.1.1)
	@echo "📈 Bumping patch version..."
	@current_version=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	new_version=$$(echo $$current_version | awk -F. '{$$3=$$3+1; print $$1"."$$2"."$$3}'); \
	sed -i.bak "s/version = \"$$current_version\"/version = \"$$new_version\"/" pyproject.toml && rm pyproject.toml.bak; \
	echo "✅ Version bumped from $$current_version to $$new_version"

version-bump-minor: ## Bump minor version (0.1.0 -> 0.2.0)
	@echo "📈 Bumping minor version..."
	@current_version=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	new_version=$$(echo $$current_version | awk -F. '{$$2=$$2+1; $$3=0; print $$1"."$$2"."$$3}'); \
	sed -i.bak "s/version = \"$$current_version\"/version = \"$$new_version\"/" pyproject.toml && rm pyproject.toml.bak; \
	echo "✅ Version bumped from $$current_version to $$new_version"

version-bump-major: ## Bump major version (0.1.0 -> 1.0.0)
	@echo "📈 Bumping major version..."
	@current_version=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	new_version=$$(echo $$current_version | awk -F. '{$$1=$$1+1; $$2=0; $$3=0; print $$1"."$$2"."$$3}'); \
	sed -i.bak "s/version = \"$$current_version\"/version = \"$$new_version\"/" pyproject.toml && rm pyproject.toml.bak; \
	echo "✅ Version bumped from $$current_version to $$new_version" 