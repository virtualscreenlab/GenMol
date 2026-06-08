import json
import matplotlib.pyplot as plt
from venn import venn

# Read the file
with open('output.txt', 'r') as f:
    content = f.read()

# Parse the three database sections
databases = {}
current_db = None
json_buffer = ""

for line in content.split('\n'):
    if line.startswith('Database '):
        if current_db and json_buffer:
            databases[current_db] = json.loads(json_buffer)
        current_db = line.strip()
        json_buffer = ""
    elif current_db:
        json_buffer += line

if current_db and json_buffer:
    databases[current_db] = json.loads(json_buffer)

# Create dictionary for venn package
smiles_dict = {}
for db_name, entries in databases.items():
    smiles_dict[db_name] = {entry['smiles'] for entry in entries}

# Create Venn diagram
plt.figure(figsize=(10, 8))
venn(smiles_dict)
plt.title('Overlap of SMILES Across Three Databases', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('venn_diagram.png', dpi=150, bbox_inches='tight')
plt.show()

# Print the SMILES in all three databases
common_all = set.intersection(*smiles_dict.values())
print("=" * 80)
print("SMILES IN ALL THREE DATABASES")
print("=" * 80)
for smiles in sorted(common_all):
    print(f"\n{smiles}:")
    for db_name in ['Database 1', 'Database 2', 'Database 3']:
        for entry in databases[db_name]:
            if entry['smiles'] == smiles:
                print(f"  {db_name}: score = {entry['score']}")