# Steps to run the Code Base

## UV installation

- On macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

- On Windows
    powershell -c "irm https://astral.sh/uv/0.1.18/install.ps1 | iex"

- Or via pip
    pip install uv

## .env file creation
-  Refer .env-example

## Activate the environment
- source .venv/bin/activate  -> On macOS/Linux
 - .venv\Scripts\activate     ->On Windows

# Run the Script
- uv run python offline/metadata_filtering.py


# I have referred this sample SQLite DB for a quick iteration
- https://www.sqlitetutorial.net/sqlite-sample-database/