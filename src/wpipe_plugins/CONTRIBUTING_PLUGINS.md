# Community Plugin Guide: Building Supported States for wpipe-plugins

Thank you for your interest in contributing to the `wpipe` ecosystem! 

**wpipe-plugins** is a community-driven repository where developers can build, share, and support their own states (steps). Unlike the [official author-maintained states](https://github.com/wisrovi/wpipe-steps), this repository is the home for community-supported innovation.

This guide establishes the professional standards required for all new community states to ensure library quality, interoperability, and long-term maintainability.

## 1. Mandatory Directory Structure

Each new state must reside within a logical namespace (e.g., `connectivity`, `vision`, `processing`) and follow this structure:

```text
src/wpipe_plugins/[category]/[state_name]/
├── __init__.py          # Exports the main state class
├── README.md            # MANDATORY: State-specific documentation
├── LICENSE              # MANDATORY: State license (MIT recommended)
├── examples/            # MANDATORY: At least one example script (.py)
├── schemas/             # Pydantic models for input/output validation
├── states/              # Core state logic (the @step class)
└── utils/               # (Optional) Helper functions
```

## 2. Mandatory Documentation per State

Each state is an independent unit. To be accepted, it must include:

### State README.md
Must contain:
- Clear description of what the state does.
- Configuration and input parameters (based on schemas).
- Quick usage example (copy-paste).
- Specific dependencies if applicable.

### State LICENSE
Even if the repository has a global license, each state must carry its own `LICENSE` file (MIT is recommended) to protect the author and clarify usage for the community.

## 3. Code Requirements

### `@step` Decorator
All states must use the `wpipe` decorator.
```python
from wpipe import step

@step(name="MyNewStep")
class MyNewStep:
    ...
```

### Typing and Validation
Using **Pydantic** for input data is mandatory. This ensures the pipeline fails early if data is incorrect.

### Logging
Do not use `print()`. Use `loguru` to maintain consistency with the rest of the library.

---

## 4. Basic Example: `rss_parser`
Location: `src/wpipe_plugins/connectivity/rss/`

This is an ideal example for states performing simple integration tasks or data processing. It focuses on a single class and straightforward validations.

---

## 5. Advanced Example: `ecam_yolo` (Vision Plugin)
Location: `src/wpipe_plugins/vision/ecam_yolo/`

Use this example if your contribution:
- Requires heavy dependencies (e.g., PyTorch).
- Has logic split across multiple files (`wrappers`, `helpers`).
- Includes multiple examples (detection, segmentation, classification).

---

## 6. Catalog Registration

To make your state visible to the community, you must add an entry to `steps_catalog.json` using the following format:

```json
{
  "name": "unique_name",
  "func_name": "ClassNameStep",
  "namespace": "wpipe_plugins.category.name",
  "version": "0.1.0",
  "description": "Short description of what this state does.",
  "license": "MIT",
  "repo": "Community",
  "author": "Your Name",
  "examples": "path/to/examples/folder/",
  "environment": "Software/Hardware requirements (e.g., Requires Redis server, GPU recommended)",
  "requirements": "path/to/plugin/requirements.txt",
  "how_to_use": "from wpipe_plugins.category.name import ClassNameStep"
}
```

## 7. Delivery Flow and Pull Requests (PR)

To ensure your contribution is properly reviewed and integrated, follow this workflow:

1.  **Work Branch:** Create your branch based on `main`.
2.  **Pull Request:** Open a PR when ready.
3.  **Target Branch:** The PR must target a branch starting with **`001***`** (e.g., `001-feature-new-state`).
4.  **Reviewers:** You must assign the **original repository author** as a PR reviewer to validate quality standards.

## 8. Delivery Checklist

- [ ] Code passes local tests (`pytest`).
- [ ] You have executed `make format` and `make lint`.
- [ ] Detailed `README.md` exists within the state folder.
- [ ] `LICENSE` file exists within the state folder.
- [ ] At least one functional example exists in the `examples/` folder.
- [ ] `steps_catalog.json` has been updated.
- [ ] The plugin's `__init__.py` exposes the main class.
