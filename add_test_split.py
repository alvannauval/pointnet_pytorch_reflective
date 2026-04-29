import json

with open('testing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for i, line in enumerate(cell.get('source', [])):
            # 1. Update the split indices definition
            if 'split_idx = int(0.8 * len(indices))' in line:
                new_source.append(line.replace('split_idx = int(0.8 * len(indices))', 'split_idx_val = int(0.8 * len(indices))\n    split_idx_test = int(0.9 * len(indices))'))
                continue
            if 'train_indices = indices[:split_idx]' in line:
                new_source.append(line.replace('[:split_idx]', '[:split_idx_val]'))
                continue
            if 'val_indices = indices[split_idx:]' in line:
                new_source.append(line.replace('[split_idx:]', '[split_idx_val:split_idx_test]\n    test_indices = indices[split_idx_test:]'))
                continue
                
            # 2. Update Dataset instantion
            if 'val_ds   = ChamferRegressionDataset' in line:
                new_source.append(line)
                new_source.append(line.replace('val_ds', 'test_ds').replace('val_indices', 'test_indices'))
                continue
                
            # 3. Update DataLoader instantiation
            if 'val_loader = DataLoader(' in line:
                new_source.append(line)
                # Keep appending the val loader block lines until its closed bracket
                # And we'll just inject test_loader right after the validation dataloader is fully initialized.
                continue
                
            # Wait, easier to look for the end of val_loader instantiation
            if 'reg_module = importlib.import_module' in line:
                # Right before initializing the model, inject test_loader
                new_source.append('    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)\n')
                new_source.append(line)
                continue
                
            # 4. Update the print statement
            if 'print(f"Training on: {len(train_ds)} | Validation: {len(val_ds)}")' in line:
                new_source.append(line.replace('}")', '} | Testing: {len(test_ds)}")'))
                continue
                
            # 5. Update Inference block to use test_ds
            if "generate_prediction_csv('./best_regression_model.pth', val_ds," in line:
                new_source.append(line.replace('val_ds', 'test_ds'))
                continue
                
            # 6. Update generate_inference_csv call
            if "generate_inference_csv(" in line and "model," in cell.get('source', [])[i+1]:
                new_source.append(line)
                continue
            if "train_indices," in line and "generate_inference_csv" not in cell.get('source', [])[i-1]:
                # We assume the user has multiline generation inference csv
                new_source.append(line)
                continue
            if "val_indices," in line and "train_indices," in cell.get('source', [])[i-1]:
                new_source.append(line)
                new_source.append(line.replace("val_indices", "test_indices"))
                continue
                
            # 7. Update generate_inference_csv definition
            if "def generate_inference_csv(model, train_indices, val_indices, metadata" in line:
                new_source.append(line.replace('val_indices, metadata', 'val_indices, test_indices, metadata'))
                continue
            if "process_split(val_indices, 'Validation')" in line:
                new_source.append(line)
                new_source.append('    print("Inference: Processing Testing Split...")\n')
                new_source.append("    process_split(test_indices, 'Testing')\n")
                continue
                
            new_source.append(line)
        cell['source'] = new_source

with open('testing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook Split Migration Complete.")
