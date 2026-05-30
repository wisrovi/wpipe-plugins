import os
import json
import re
import tomli

def get_author(repo_path):
    pyproject_path = os.path.join(repo_path, "pyproject.toml")
    if os.path.exists(pyproject_path):
        try:
            with open(pyproject_path, "rb") as f:
                data = tomli.load(f)
                authors = data.get("project", {}).get("authors", [])
                if authors:
                    return authors[0].get("name", "Unknown Author")
        except Exception: pass
    return "Community Author"

def scan_repo(repo_root, prefix="Official"):
    catalog = []
    repo_author = get_author(repo_root)
    # The actual code is in wpipe_steps or module
    code_path = os.path.join(repo_root, "wpipe_steps" if prefix == "Official" else "module")
    if not os.path.exists(code_path):
        return catalog
        
    base_dir = os.path.dirname(code_path)
    
    for root, dirs, files in os.walk(code_path):
        if any(x in root for x in ["node_modules", ".git", "examples", "__pycache__"]):
            continue
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        matches = re.finditer(r'@step\s*\((.*?)\)', content, re.DOTALL)
                        for m in matches:
                            args_text = m.group(1)
                            name_match = re.search(r'name\s*=\s*["\'](.*?)["\']', args_text)
                            if not name_match: continue
                            name = name_match.group(1)
                            
                            after_decorator = content[m.end():]
                            func_match = re.search(r'(?:def|class)\s+(\w+)', after_decorator)
                            func_name = func_match.group(1) if func_match else ""
                            
                            rel_to_base = os.path.relpath(root, base_dir)
                            namespace = rel_to_base.replace(os.sep, ".")
                            if file != "__init__.py":
                                namespace = f"{namespace}.{file[:-3]}"
                            
                            catalog.append({
                                "name": name,
                                "func_name": func_name,
                                "namespace": namespace,
                                "repo": prefix,
                                "author": repo_author,
                                "file": os.path.relpath(file_path, base_dir)
                            })
                except Exception: pass
    return catalog

if __name__ == "__main__":
    repo_root = os.getcwd()
    if os.path.exists("wpipe_steps"):
        cat = scan_repo(repo_root, "Official")
    elif os.path.exists("module"):
        cat = scan_repo(repo_root, "Community")
    else:
        cat = []
        
    with open("steps_catalog.json", "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=2)
    print(f"steps_catalog.json generated with {len(cat)} entries for author: {get_author(repo_root)}")
