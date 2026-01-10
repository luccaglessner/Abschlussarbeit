import json
import os

# Define path
file_path = "4_Imputation/4.1_VAE_Imputation/VAE_Imputation.ipynb"
full_path = os.path.abspath(file_path)

print(f"Reading file: {full_path}")
try:
    with open(full_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except FileNotFoundError:
    print(f"File not found: {full_path}")
    exit(1)

# Target line to replace
target_line_content = "for i, combo in enumerate(combinations):"
cell_found = False

print("Searching for target cell...")
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this cell contains our target loop
        if any(target_line_content in line for line in source):
             print("Cell found!")
             cell_found = True
             
             final_source = []
             for line in source:
                 if target_line_content in line:
                     print("Injecting prioritization logic...")
                     # Insering the new logic replacing the simple loop
                     final_source.extend([
                        "\n",
                        "    # ------------------------- Priorisierung nach Datenmenge -------------------------\n",
                        "    print(\"\\nBerechne Datenverfügbarkeit für Priorisierung...\")\n",
                        "    possible_runs = []\n",
                        "    for combo in combinations:\n",
                        "        # Temporäre Feature-Ermittlung für Count\n",
                        "        current_selection = FIXED_BASE_FEATURES + list(combo)\n",
                        "        t_cols, _ = get_training_features(current_selection, df_full.columns)\n",
                        "        n_rows = df_full[t_cols].dropna().shape[0]\n",
                        "        possible_runs.append((n_rows, combo))\n",
                        "\n",
                        "    # Sortieren: Meiste Daten zuerst (descending)\n",
                        "    possible_runs.sort(key=lambda x: x[0], reverse=True)\n",
                        "\n",
                        "    # Base Run (leere Combo) an die erste Stelle setzen\n",
                        "    base_run_entry = next((item for item in possible_runs if len(item[1]) == 0), None)\n",
                        "    if base_run_entry:\n",
                        "        possible_runs.remove(base_run_entry)\n",
                        "        possible_runs.insert(0, base_run_entry)\n",
                        "\n",
                        "    print(\"Top 5 Runs nach Datenmenge:\")\n",
                        "    for k in range(min(5, len(possible_runs))):\n",
                        "        print(f\"  {k+1}. Rows={possible_runs[k][0]} | Combo={possible_runs[k][1]}\")\n",
                        "\n",
                        "    # ------------------------- Loop über sortierte Runs -------------------------\n",
                        "    for i, (n_rows_preview, combo) in enumerate(possible_runs):\n"
                     ])
                 else:
                     final_source.append(line)
             
             cell['source'] = final_source
             break

if cell_found:
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Target cell not found or logic already applied.")
