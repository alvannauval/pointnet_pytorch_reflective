import json
import re

with open('testing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    
    source = cell['source']
    new_source = []
    
    for line in source:
        # Standardize Inference / Forward Pass unpacks
        if 'pred, _ = model(points)' in line:
            line = line.replace('pred, _ = model(points)', 'pred, trans_feat = model(points)')
            # If it had a # PointNet++ comment, remove it to show it's unified
            line = line.replace(' # PointNet++', '')
            
        # Standardize Loss Computations
        if 'loss = criterion(pred, target, None)' in line:
            line = line.replace('loss = criterion(pred, target, None)', 'loss = criterion(pred, target, trans_feat)')
            line = line.replace(' # PointNet++', '')
            
        if 'v_loss = criterion(pred, target, None)' in line:
            line = line.replace('v_loss = criterion(pred, target, None)', 'v_loss = criterion(pred, target, trans_feat)')
            
        new_source.append(line)
        
    cell['source'] = new_source

with open('testing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook Standardized Successfully!")
