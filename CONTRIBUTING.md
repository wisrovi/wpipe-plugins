# Contributing to wpipe

Thank you for your interest in contributing to wpipe!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/wpipe-plugins.git](https://github.com/YOUR_USERNAME/wpipe-plugins.git)
   cd wpipe-plugins
   ```
3. Configure the upstream remote to keep your fork synced with the main repository:
   ```bash
   git remote add upstream [https://github.com/wisrovi/wpipe-plugins.git](https://github.com/wisrovi/wpipe-plugins.git)
   ```
4. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Always pull the latest changes from the main repository before working:
   ```bash
   git checkout main
   git pull upstream main
   ```
2. Create a new branch for your feature, bugfix, or new plugin:
   ```bash
   git checkout -b feature/your-plugin-name
   ```
3. Make your changes, ensure code quality, and commit them:
   ```bash
   git add .
   git commit -m "feat: add your awesome plugin"
   ```
4. Push the branch to your own GitHub fork:
   ```bash
   git push origin feature/your-plugin-name
   ```
5. Go to the original `wpipe-plugins` repository on GitHub, and you will see a banner to open a Pull Request from your branch.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=wpipe_plugins --cov-report=html
```

### Code Quality

```bash
# Lint with ruff
ruff check wpipe_plugins/

# Type checking with mypy
mypy wpipe_plugins/

# Format code with black
black wpipe_plugins/
```

### Running All Quality Checks

```bash
ruff check wpipe_plugins/ && mypy wpipe_plugins/ && pytest
```

## Writing Tests

- All new plugins or features must include tests
- Tests should cover plugin initialization and core execution logic
- Use descriptive test names: test_<plugin_name>_<behavior>



## Pull Request Guidelines

1. Ensure all tests pass
2. Run linting: ruff check wpipe_plugins/
3. Run type checking: mypy wpipe_plugins/
4. Update documentation or README if needed (add your plugin description)
5. Keep changes focused and atomic
6. Make sure you are submitting the PR from your fork's branch to wpipe-plugins:main

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to public functions
- Keep functions small and focused

## Reporting Issues

- Use the GitHub issue tracker
- Include a minimal reproducible example
- Specify your Python version and OS

## Author
- William Rodríguez - wisrovi

