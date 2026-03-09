import json
from pathlib import Path

def convert_notebook_to_python(notebook_path) -> str:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_lines = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for line in source:
                if line.strip().startswith('%') or line.strip().startswith('!'):
                    line = '# ' + line
                code_lines.append(line)
            code_lines.append('\n\n')
            
    return ''.join(code_lines)

nb_path = Path(r"f:\Abschlussarbeit\Abschlussarbeit Bearbeitung\Jupyter Notebooks\4_Imputation\4.3_Evaluation\Evaluation.ipynb")
print(f"Converting {nb_path}...")
code = convert_notebook_to_python(nb_path)
print(f"Conversion successful. Code length: {len(code)}")
print("First 100 chars:")
print(code[:100])
