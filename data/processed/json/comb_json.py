import json

FIL1 = './data/processed/json/ds3_trn.json'
FIL2 = './data/processed/json/ds3_tst.json'

# Load the first JSON
with open(file=FIL1, mode='r', encoding='utf-8') as f1:
    data1 = json.load(f1)

# Load the second JSON
with open(file=FIL2, mode='r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# Append annotations from the second JSON to the first
data1['annotations'].extend(data2['annotations'])

# Save the combined JSON to a new file
with open(file='ds3_an.json', mode='w', encoding='utf-8') as f_combined:
    json.dump(data1, f_combined, indent=4)
