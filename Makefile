.PHONY: install test demo

install:
	pip install -e ".[dev]"

test:
	pytest -q

demo:
	python -m agent_prompt_versioning.demo
