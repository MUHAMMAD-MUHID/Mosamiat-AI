# Step 01: Local development environment

Prepare and verify the environment only. Application code, data-source selection,
cloud setup, and deployment belong to later reviewed steps.

## Installed foundation

- Python 3.11.18 in `.tools/python`, isolated from system Python.
- uv 0.13.8 in `.tools/bootstrap` for environment and dependency management.
- `.venv` with HTTPX for API requests and pandas for data inspection.
- pytest for future tests and Ruff for linting and formatting.
- `uv.lock` records exact dependency versions. Keep it with `pyproject.toml`.
- VS Code's default project interpreter points to `.venv/Scripts/python.exe`.

Python 3.11 is the initial compatibility baseline. Hopsworks, modeling, agents,
and dashboard dependencies will be checked and added when their milestones begin.
This step does not establish compatibility for the entire future application.

## Recreate the environment on Windows

Run these PowerShell commands from the project root. The bootstrap requires an
existing Python installation with pip and internet access.

```powershell
python -m pip --isolated install --index-url https://pypi.org/simple --target .tools/bootstrap uv==0.13.8
& ./.tools/bootstrap/bin/uv.exe python install 3.11.18 --install-dir .tools/python --no-bin --no-registry
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.tools/python'
& ./.tools/bootstrap/bin/uv.exe sync --locked
```

The Python install command uses Astral's managed Python distribution and leaves
system Python, PATH, and Windows registry registration unchanged. See the
[uv Python guide](https://docs.astral.sh/uv/guides/install-python/).

## Daily commands

Activation is optional. These commands use the project environment directly:

```powershell
& ./.venv/Scripts/python.exe --version
& ./.venv/Scripts/ruff.exe check .
& ./.venv/Scripts/pytest.exe --version
& ./.tools/bootstrap/bin/uv.exe pip check --python .venv/Scripts/python.exe
```

Once application tests exist, run `& ./.venv/Scripts/pytest.exe`.
Until then, pytest has no tests to run; that is not a passing application test suite.
In VS Code, select `.venv/Scripts/python.exe` if an interpreter was already selected
before these workspace settings were added.

Add a dependency only when the current milestone needs it, using `uv add` through
the local executable (`uv add --dev` for development tools). Review the resulting
changes to `pyproject.toml` and `uv.lock` together.

## Local-only files

`.gitignore` excludes all four supplied reference files, `.venv`, `.tools`, caches,
secrets, and generated data. The supplied roadmap remains a local reference.
No API keys are required for this environment setup. No commits or pushes are
part of this step.

## Completion checks

- The project interpreter runs Python 3.11.18 from `.venv`.
- HTTPX, pandas, NumPy, pytest, and Ruff are available in that environment.
- `uv sync --locked --offline` can reproduce the installed dependency set.
- `uv pip check` reports no dependency conflicts.
- Git ignores every supplied reference file and local runtime directory.

Next milestone: review free historical weather and air-quality data for Islamabad.
