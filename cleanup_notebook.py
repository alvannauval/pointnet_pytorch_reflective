import json
import re

with open('testing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    
    source = cell['source']
    new_source = []
    
    skip_mode = False
    
    for i, line in enumerate(source):
        # --- DATASET CHANGES ---
        if 'def __init__(self, metadata, pcd_dir, indices, num_points=1024' in line:
            new_source.append(line.split(', is_training')[0] + ', is_training=False):\n')
            continue
        if 'self.y_mean = y_mean' in line or 'self.y_std = y_std' in line or 'self.use_norm = use_norm' in line:
            continue
        if 'chamfer_out = (chamfer - self.y_mean) / self.y_std if self.use_norm else chamfer' in line:
            new_source.append(line.replace('chamfer_out = (chamfer - self.y_mean) / self.y_std if self.use_norm else chamfer', 'chamfer_out = chamfer'))
            continue
            
        # --- TRAINING LOOP CHANGES ---
        if 'def run_training(' in line and 'use_loss_weighting' in line:
            line = re.sub(r',\s*y_mean=0\.0,\s*y_std=1\.0,\s*use_norm=[^,]+,\s*use_loss_weighting=[^,]+', '', line)
            new_source.append(line)
            continue
            
        # Delete if use_loss_weighting block
        if '# --- NEW: Inverse Frequency Weighting ---' in line:
            skip_mode = True
            continue
        if skip_mode and '# ----------------------------------------' in line:
            skip_mode = False
            continue
        if skip_mode:
            continue
            
        if 'if use_norm:' in line:
            # Need to skip this and the next line
            source[i+1] = "" # Blank it out so it doesn't get appended next iteration
            continue
            
        # --- MAIN CHANGES ---
        if 'USE_NORMALIZATION =' in line or 'USE_LOSS_WEIGHTING =' in line:
            continue
            
        if 'is_training=' in line and 'y_mean=y_mean' in line:
            line = re.sub(r',\s*y_mean=y_mean,\s*y_std=y_std,\s*use_norm=USE_NORMALIZATION', '', line)
            
        if 'train_history = run_training(' in line or 'model, train_loader, val_loader' in line:
            line = re.sub(r',\s*y_mean=y_mean,\s*y_std=y_std,\s*use_norm=USE_NORMALIZATION,\s*use_loss_weighting=USE_LOSS_WEIGHTING', '', line)
            
        # --- INFERENCE CSV CHANGES ---
        if 'def generate_prediction_csv(' in line:
            line = re.sub(r',\s*use_norm=[^,]+,\s*y_mean=0\.0,\s*y_std=1\.0', '', line)
        if 'output_file=' in line and 'use_norm=USE_NORMALIZATION' in line:
            line = re.sub(r',\s*use_norm=USE_NORMALIZATION,\s*y_mean=y_mean,\s*y_std=y_std', '', line)
            
        if 'def generate_inference_csv(' in line:
            line = re.sub(r',\s*y_mean,\s*y_std,\s*use_norm', '', line)
        if 'generate_inference_csv(' in line and 'USE_NORMALIZATION' in source[min(i+8, len(source)-1)]:
            # This is the function call which is multiline
            pass
        if 'y_mean,' in line and 'y_std,' in source[min(i+1, len(source)-1)]:
            continue
        if 'y_std,' in line and 'USE_NORMALIZATION' in source[min(i+1, len(source)-1)]:
            continue
        if 'USE_NORMALIZATION' in line and ')' in line and 'y_std,' in source[max(i-1, 0)]:
            new_source.append('    )\n')
            continue
            
        # Inside inference functions (process_split)
        if 'y_mean=y_mean, y_std=y_std,' in line:
            continue
        if 'use_norm=use_norm' in line:
            continue
        if 'if use_norm:' in line and 'pred_final = ' in source[min(i+1, len(source)-1)]:
            # skip the if/else entirely and just do the assignments
            source[i+1] = ""
            source[i+2] = ""
            source[i+3] = ""
            source[i+4] = ""
            source[i+5] = ""
            new_source.append(line.replace('if use_norm:', 'pred_final = pred_norm\n'))
            new_source.append(line.replace('if use_norm:', '            target_final = target_val\n'))
            continue
            
        if 'if USE_NORMALIZATION:' in line and 'pred_final = ' in source[min(i+1, len(source)-1)]:
            source[i+1] = ""
            source[i+2] = ""
            source[i+3] = ""
            source[i+4] = ""
            source[i+5] = ""
            new_source.append(line.replace('if USE_NORMALIZATION:', 'pred_final = pred_norm\n'))
            new_source.append(line.replace('if USE_NORMALIZATION:', 'target_final = target.numpy()[0]\n'))
            continue
            
        if 'if USE_NORMALIZATION:' in line and 't_val = ' in source[min(i+1, len(source)-1)]:
            source[i+1] = ""
            continue
            
        # Cell 5 Inference Process Split unnormalize
        if 'if use_norm:' in line and 'pred_val = ' in source[min(i+1, len(source)-1)]:
            source[i+1] = ""
            source[i+2] = ""
            source[i+3] = ""
            source[i+4] = ""
            source[i+5] = ""
            new_source.append(line.replace('if use_norm:', 'pred_val = pred.item()\n'))
            new_source.append(line.replace('if use_norm:', '                    gt_val = target.item()\n'))
            continue

        if line != "":
            new_source.append(line)
            
    cell['source'] = new_source

with open('testing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Cleanup script executed perfectly!")
