# wpipe-plugins

Plugins and extensions for the [wpipe](https://github.com/wisrovi/wpipe) ecosystem.

## Overview

This repository contains a collection of professional plugins (steps) for the `wpipe` pipeline engine, ranging from AI vision models to connectivity tools.

## Key Features

- **Standardized Structure**: Every plugin follows a strict professional layout.
- **Auto-Cataloging**: Integrated `steps_catalog.json` for easy discovery.
- **Validation-First**: All plugins use Pydantic for robust input validation.
- **Docker-Ready**: Standardized testing environment using Docker.

## Quick Start

### Installation

```bash
pip install wpipe-plugins
```

### Usage Example (RSS Parser)

```python
from wpipe_plugins.connectivity.rss import RSSParserStep
from wpipe import Pipeline

# Initialize the step
rss_step = RSSParserStep()

# Create a pipeline
pipe = Pipeline(pipeline_name="news_fetcher")
pipe.set_steps([rss_step])

# Run
results = pipe.run({"url": "https://news.google.com/rss", "limit": 3})
print(results)
```

## Contributing

We welcome community contributions! Please read our [Plugin Contribution Guide](src/wpipe_plugins/CONTRIBUTING_PLUGINS.md) to get started.

Contributions are welcome! Please follow these steps to contribute to the project:

1. **Fork** the repository.
2. **Clone** your fork to your local machine.
3. Create a new **branch** for your feature or bugfix (`git checkout -b feature/amazing-feature`).
4. **Commit** your changes (`git commit -m 'Add some amazing feature'`).
5. **Push** to the branch (`git push origin feature/amazing-feature`).
6. Open a **Pull Request** against the main repository.

### Contribution Rules
- Target your Pull Request to a branch starting with `001***`.
- Assign @william-rodriguez as a mandatory reviewer.
- Ensure your plugin includes a `README.md`, `LICENSE`, and `examples/`.

Please ensure your code adheres to the project's standards and all tests pass before submitting.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**William Steve Rodriguez Villamizar**
- LinkedIn: [william-steve-rodriguez](https://www.linkedin.com/in/william-steve-rodriguez/)
- Email: wisrovi.rodriguez@gmail.com
