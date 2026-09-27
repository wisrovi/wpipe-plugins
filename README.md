<p align="center">
  <a href="https://github.com/wisrovi/wpipe-plugins"><img src="https://img.shields.io/badge/wpipe--plugins-v1.5.3-blue.svg?style=for-the-badge&logo=python&color=3b82f6" alt="wpipe-plugins" /></a>
  <a href="https://linkedin.com/in/wisrovi-rodriguez"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  <a href="https://wisrovi.dev"><img src="https://img.shields.io/badge/Author-wisrovi.dev-111827?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Portal" /></a>
  <a href="https://orcid.org/0009-0005-0710-1861"><img src="https://img.shields.io/badge/ORCID-0009--0005--0710--1861-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License" /></a>
</p>

# 🔌 WPipe-Plugins — Dynamic Community Plugins Framework for WPipe

A dynamic plugin and step-extension framework for the **WPipe** orchestration engine. Allows developers and community contributors to publish, discover, and dynamically mount custom steps and states into distributed execution graphs.

> **Important Note**: This repository is a community-driven space designed for the community to build, share, and support their own states (steps) for `wpipe`. If you are looking for official states maintained directly by the author, visit [wpipe-steps](https://github.com/wisrovi/wpipe-steps).

```mermaid
flowchart LR
    A["Community Contributor / Plugin Developer"] --> B["Module Scaffolding (@step)"]
    B --> C["WPipe Plugins Catalog Scanner"]
    C --> D["Dynamic Registry (JSON / MCP)"]
    D --> E["WPipe Core Engine & Workers"]
    style A fill:#1e293b,stroke:#3b82f6,color:#ffffff
    style B fill:#1e293b,stroke:#10b981,color:#ffffff
    style C fill:#1e293b,stroke:#f59e0b,color:#ffffff
    style D fill:#1e293b,stroke:#ec4899,color:#ffffff
    style E fill:#1e293b,stroke:#8b5cf6,color:#ffffff
```

---

## 🌟 Key Features

- **🔌 Plug-and-Play Extensibility**: Develop custom steps decorated with `@step(name="...")` and easily distribute them.
- **🔍 Automated Catalog Generation**: Built-in AST and pattern scanner (`generate_catalog.py`) parses community steps and namespaces into `steps_catalog.json`.
- **🤝 WPipe-MCP Integration**: Steps indexed here become automatically discoverable by agentic workflows in `wpipe-mcp`.
- **🛡️ Strict Typing & Schemas**: Enforces Pydantic / dataclass contexts and clean separation of concerns.
- **🐳 Docker-Ready**: Standardized testing and execution environment.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/wisrovi/wpipe-plugins.git
cd wpipe-plugins
pip install -e .
```

### 2. Usage Example (RSS Parser)

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

### 3. Scanning & Generating Catalog

```bash
python generate_catalog.py
```

---

## 📂 Project Structure

- `module/`: Custom plugin definitions and community step implementations.
- `examples/`: Reference implementations and step configurations.
- `generate_catalog.py`: AST scanner that generates the JSON registry of available plugins.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-plugin`).
3. Commit your changes (`git commit -m 'feat: add amazing plugin'`).
4. Push to branch and open a Pull Request against branch `001-DEVELOPMENT`.

---

## 📄 License

MIT License

---

## 👤 Autor & Afiliación Oficial

* **William Steve Rodriguez Villamizar (Wisrovi)**
* **Cargo:** Principal AI Engineer & Applied AI Solutions Architect | Scientific Researcher
* 📧 **Email:** [wisrovi.rodriguez@gmail.com](mailto:wisrovi.rodriguez@gmail.com)
* 🌐 **Portal Oficial:** [wisrovi.dev](https://wisrovi.dev)
* 💼 **LinkedIn:** [wisrovi-rodriguez](https://www.linkedin.com/in/wisrovi-rodriguez/)
* 🆔 **ORCID:** [0009-0005-0710-1861](https://orcid.org/0009-0005-0710-1861)
* 📦 **PyPI:** [pypi.org/user/wisrovi/](https://pypi.org/user/wisrovi/)
* 🐙 **GitHub:** [@wisrovi](https://github.com/wisrovi)
