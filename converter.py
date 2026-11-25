import os
import json
import glob

# Папка с notebooks — смени ако е различна
input_folder  = "."
output_folder = "py_scripts"

# Създаваме изходната папка ако не съществува
os.makedirs(output_folder, exist_ok=True)

# Намираме всички .ipynb файлове
notebooks = glob.glob(os.path.join(input_folder, "*.ipynb"))

if not notebooks:
    print("Не са намерени .ipynb файлове в текущата папка.")
else:
    for nb_path in sorted(notebooks):
        nb_name = os.path.basename(nb_path)
        py_name = os.path.splitext(nb_name)[0] + ".py"
        py_path = os.path.join(output_folder, py_name)

        with open(nb_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        code_lines = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                if source.strip():
                    code_lines.append(source)
                    code_lines.append("\n\n")

        with open(py_path, "w", encoding="utf-8") as f:
            f.write("".join(code_lines))

        print(f"{nb_name} → {py_path}")

print(f"\nГотово! Файловете са в папка: {output_folder}/")