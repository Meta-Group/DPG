Test Runner Notes

Local, no-install run (uses repo root on PYTHONPATH):

    PYTHONPATH=. pytest -q

Run a single file:

    PYTHONPATH=. pytest -q tests/test_integration_dpg.py

Run only plot-related tests:

    PYTHONPATH=. pytest -q -k plot

Notebook tests (optional):

    pip install nbmake
    PYTHONPATH=. pytest --nbmake tutorials/*.ipynb

Alternative using nbconvert:

    jupyter nbconvert --to notebook --execute tutorials/*.ipynb --output-dir /tmp/nb-out
