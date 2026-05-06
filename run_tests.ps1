# Run the test suite using the project venv, not the system Python.
# Usage:
#   .\run_tests.ps1              # all tests
#   .\run_tests.ps1 tests/unit   # unit tests only
#   .\run_tests.ps1 tests/e2e    # E2E tests only (needs playwright installed)
#   .\run_tests.ps1 -v           # verbose

$pytest = ".\.venv\Scripts\pytest.exe"
if (-not (Test-Path $pytest)) {
    Write-Error "Venv not found. Run: python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt"
    exit 1
}

& $pytest @args
