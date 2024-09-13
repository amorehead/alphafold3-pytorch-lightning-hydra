#!/bin/bash
cp .env.example .env
pip install uv
uv pip install -e '.[test]'
pytest tests/
