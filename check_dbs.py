import json
import re
import os

nb_path = r'F:\Abschlussarbeit\Abschlussarbeit Bearbeitung\Jupyter Notebooks\1_Acquisition\1.1_Data-Acquisition-Wrapper\Data-Acquisition-Wrapper.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

text = ' '.join([''.join(c['source']) for c in data['cells'] if c.get('cell_type') == 'code'])

# Find all file names ending in .xlsx, .xls, .csv
used = set(re.findall(r'[\w\.-]+\.(?:xlsx|xls|csv)', text))
print('USED IN NOTEBOOK:')
for u in sorted(used):
    print(f'  - {u}')

bases = [
    r'F:\Abschlussarbeit\Abschlussarbeit Bearbeitung\Jupyter Notebooks\1_Acquisition\1.1_Data-Acquisition-Wrapper\Gesammelte_Datenbanken',
    r'F:\Abschlussarbeit\Abschlussarbeit Bearbeitung\Jupyter Notebooks\1_Acquisition\1.1_Data-Acquisition-Wrapper\Angepasste_Datenbanken'
]

existing = set()
for base in bases:
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith(('.xlsx', '.xls', '.csv')) and not file.startswith('~'):
                existing.add(file)

print('\nEXISTING IN FOLDERS:')
for e in sorted(existing):
    print(f'  - {e}')

not_used = existing - used
print('\nNOT USED:')
for n in sorted(not_used):
    print(f'  - {n}')

used_not_found = used - existing
print('\nUSED BUT NOT FOUND:')
for u in sorted(used_not_found):
    print(f'  - {u}')
