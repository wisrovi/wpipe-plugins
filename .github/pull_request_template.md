## 🚀 New State / Plugin Proponent

**State Name:** 
**Author:** @
**Category:** (e.g., vision, connectivity, processing)

---

### 📝 Description
Short description of the functionality this new state brings to the `wpipe` ecosystem.

---

### ✅ Minimum Viable Standard (MVS) Checklist
*Please mark completed items with an `x`:*

- [ ] **Structure:** I have followed the directory structure `src/wpipe_plugins/[category]/[name]/`.
- [ ] **Core:** The main class uses the `wpipe` `@step` decorator.
- [ ] **Validation:** I use `Pydantic` to validate input data in `schemas/`.
- [ ] **Local Documentation:** I have included a specific `README.md` within the state folder.
- [ ] **Legal:** I have included a `LICENSE` file (MIT recommended) within the state folder.
- [ ] **Examples:** I have included at least one functional script in `examples/`.
- [ ] **Catalog:** I have added the corresponding entry in `steps_catalog.json` with all fields (`namespace`, `version`, `how_to_use`, `requirements`, etc.).
- [ ] **Quality:** I have executed `make format` and `make lint` locally with no errors.
- [ ] **Target:** This PR targets a `001***` branch.

---

### 🛠 Technical Requirements
What does the reviewer need to install to test this state?
*(e.g., pip install ultralytics, requires internet connection, etc.)*

---

### 🧪 Test Evidence
Paste the output of the example execution or a screenshot if applicable:

```bash
# Execution example
python src/wpipe_plugins/.../examples/example.py
```

---

**CC:** @william-rodriguez (Original Author / Mandatory Reviewer)
