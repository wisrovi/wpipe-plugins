import os
import json
import re

def scan_repo(repo_path, prefix="Official"):
    catalog = []
    base_dir = os.path.dirname(repo_path)
    
    for root, dirs, files in os.walk(repo_path):
        if "node_modules" in root or ".git" in root or "examples" in root or "__pycache__" in root:
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
                            
                            # Real import namespace
                            rel_to_base = os.path.relpath(root, base_dir)
                            namespace = rel_to_base.replace(os.sep, ".")
                            
                            # If it is not in __init__.py, append filename to namespace for direct import
                            if file != "__init__.py":
                                module_name = file[:-3]
                                namespace = f"{namespace}.{module_name}"
                            
                            catalog.append({
                                "name": name,
                                "func_name": func_name,
                                "namespace": namespace,
                                "repo": prefix,
                                "file": os.path.relpath(file_path, base_dir)
                            })
                except Exception: pass
    return catalog

if __name__ == "__main__":
    if os.path.exists("wpipe_steps"):
        cat = scan_repo("wpipe_steps", "Official")
    elif os.path.exists("module"):
        cat = scan_repo("module", "Community")
    else:
        cat = []
        
    with open("steps_catalog.json", "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=2)
    print(f"steps_catalog.json generated with {len(cat)} entries.")
