install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-training:
	python -m pip install -r requirements-training.txt

install-test: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

test: ## [Local development] Run unit tests
	python -m pytest -x -s -v tests
