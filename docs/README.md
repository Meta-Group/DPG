# DPG Documentation

This directory contains the Sphinx-based documentation for DPG.

## Build and Serve Locally

### Prerequisites

Install the documentation dependencies:

```bash
pip install ".[docs]"
```

This installs:
- `sphinx` - Documentation generator
- `pydata-sphinx-theme` - Theme (same as NumPy, pandas, scikit-learn)
- `sphinx-autoapi` - Auto-generate API docs from docstrings
- `myst-parser` - Markdown support
- `sphinx-copybutton` - Copy button for code blocks
- `sphinx-design` - Grid cards, tabs, badges

### Build the Documentation

```bash
sphinx-build -b html docs/ docs/_build/html
```

### Serve Locally

**Option 1: Python HTTP Server (Recommended)**
```bash
cd docs/_build/html
python3 -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

**VS Code Remote SSH:** VS Code will automatically forward port 8000 and provide a clickable link in the terminal output or the **Ports** panel.

**Option 2: Open directly in browser**
```bash
# Linux
xdg-open docs/_build/html/index.html

# macOS
open docs/_build/html/index.html

# Windows
start docs/_build/html/index.html
```

**Option 3: Sphinx Autobuild (Auto-reload on changes)**
```bash
pip install sphinx-autobuild
sphinx-autobuild docs/ docs/_build/html --open-browser
```

## Documentation Structure

- `index.md` - Main landing page
- `quickstart.md` - Quick start guide
- `conf.py` - Sphinx configuration
- `_static/` - Custom CSS, JS, images
- `_build/` - Generated HTML (git-ignored)

## API Reference

The API documentation is auto-generated from docstrings using `sphinx-autoapi`. It scans:
- `dpg/` - Main DPG package
- `metrics/` - Metrics package
- `counterfactual/` - Counterfactual explanations

No need to manually write API docs - just add proper docstrings to your Python code!

## Publishing

Documentation is automatically built and deployed to ReadTheDocs via the CI/CD pipeline (see `.github/workflows/docs.yml`) when changes are merged to the `main` branch.

## Troubleshooting

**Warning: "Unknown type: placeholder"**
This is a known benign warning from sphinx-autoapi when it encounters C-extension types. It's filtered out in the CI build but you may see it locally. Safe to ignore.

**Module not found errors**
Make sure you've installed the package:
```bash
pip install -e .
```

**Graphviz not found (for diagrams)**
The documentation doesn't require Graphviz unless you're embedding diagrams. If needed:
- Ubuntu/Debian: `sudo apt-get install graphviz`
- macOS: `brew install graphviz`
- Windows: `winget install Graphviz.Graphviz`

Note: Graphviz is NOT required for basic doc builds.
