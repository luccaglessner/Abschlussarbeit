import json
import os

file_path = "4_Imputation/4.1_VAE_Imputation/VAE_Imputation.ipynb"
full_path = os.path.abspath(file_path)

print(f"Fixing indentation in: {full_path}")

try:
    with open(full_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Marker string to identify the start of our injected block
start_marker = "# ------------------------- Priorisierung nach Datenmenge -------------------------"
# Marker string to identify the end of our injected block (the loop header)
end_marker = "for i, (n_rows_preview, combo) in enumerate(possible_runs):"

cell_found = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this cell contains our injected code
        if any(start_marker in line for line in source):
             print("Target cell found.")
             cell_found = True
             
             new_source = []
             inside_block = False
             
             for line in source:
                 # Check start of block
                 if start_marker in line:
                     inside_block = True
                 
                 if inside_block:
                     # Remove exactly 4 spaces from the start if possible
                     if line.startswith("    "):
                         new_line = line[4:]
                     else:
                         new_line = line # Should not happen based on my previous script, but safety first
                     
                     new_source.append(new_line)
                     
                     # Check end of block
                     if end_marker in line:
                         inside_block = False
                 else:
                     new_source.append(line)
             
             cell['source'] = new_source
             break

if cell_found:
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Indentation fixed successfully.")
else:
    print("Could not find the target code block to fix.")
