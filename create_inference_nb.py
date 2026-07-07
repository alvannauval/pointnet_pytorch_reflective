import json
import os

source_file = r"c:\Users\Alvan\Documents\Alvan\Data\Code\Python\Main\pointnet_pytorch_reflective\1_training.ipynb"
target_file = r"c:\Users\Alvan\Documents\Alvan\Data\Code\Python\Main\pointnet_pytorch_reflective\2_inference.ipynb"

# 1. Extract ChamferRegressionDataset from 1_training.ipynb
with open(source_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

dataset_lines = []
found_class = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for line in source:
            if line.startswith("class ChamferRegressionDataset"):
                found_class = True
            
            if found_class:
                dataset_lines.append(line)
                if line.startswith("# =========================================="):
                    # End of class definition
                    dataset_lines.pop() # remove the banner line
                    found_class = False
                    break
        if dataset_lines:
            break

# 2. Construct New Notebook Cells

cell1_source = [
    "# SETUP & DATASET\n",
    "\n",
    "import os\n",
    "import sys\n",
    "import torch\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "import importlib\n",
    "import open3d as o3d\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "BASE_DIR = os.getcwd()\n",
    "sys.path.append(os.path.join(BASE_DIR, 'models'))\n",
    "\n"
] + dataset_lines

cell2_source = [
    "# INFERENCE CSV #\n",
    "\n",
    "def generate_inference_csv(model_path, metadata_path, pcd_dir, device):\n",
    "    print(f\"Loading metadata from {metadata_path}...\")\n",
    "    metadata = pd.read_csv(metadata_path)\n",
    "    \n",
    "    # Create indices for ALL rows in the metadata\n",
    "    all_indices = list(range(len(metadata)))\n",
    "    \n",
    "    # Initialize the dataset\n",
    "    ds = ChamferRegressionDataset(\n",
    "        metadata, pcd_dir, all_indices, num_points=5000,\n",
    "        is_training=False, \n",
    "    )\n",
    "    # Batch size of 16 and multiprocessing to speed up inference\n",
    "    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)\n",
    "    \n",
    "    # Load model architecture and weights\n",
    "    print(f\"Loading model weights from {model_path}...\")\n",
    "    reg_module = importlib.import_module('pointnet2_regression')\n",
    "    model = reg_module.get_model(normal_channel=True).to(device)\n",
    "    model.load_state_dict(torch.load(model_path))\n",
    "    model.eval()\n",
    "    \n",
    "    results = []\n",
    "    \n",
    "    print(f\"Running inference on {len(all_indices)} samples...\")\n",
    "    with torch.no_grad():\n",
    "        for i, (points, target) in enumerate(loader):\n",
    "            points = points.to(device)\n",
    "            pred, trans_feat = model(points)\n",
    "            \n",
    "            for b in range(points.size(0)):\n",
    "                pred_val = pred[b].item()\n",
    "                gt_val = target[b].item()\n",
    "                error = abs(gt_val - pred_val)\n",
    "                \n",
    "                global_idx = all_indices[i * 16 + b]\n",
    "                filename = metadata.iloc[global_idx]['filename']\n",
    "                \n",
    "                results.append({\n",
    "                    'Original_Index': global_idx,\n",
    "                    'Filename': filename,\n",
    "                    'Split': 'Testing',\n",
    "                    'GroundTruth_CD': gt_val,\n",
    "                    'Predicted_CD': pred_val,\n",
    "                    'Error': error\n",
    "                })\n",
    "\n",
    "    # Create DataFrame and Save\n",
    "    df_results = pd.DataFrame(results)\n",
    "    df_results = df_results.sort_values(by=\"Original_Index\", ascending=True)\n",
    "    \n",
    "    save_path = \"inference_results.csv\"\n",
    "    df_results.to_csv(save_path, index=False)\n",
    "    print(f\"Results saved to {save_path}\")\n",
    "\n",
    "# ==========================================\n",
    "# EXECUTION\n",
    "# ==========================================\n",
    "if __name__ == \"__main__\":\n",
    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "    \n",
    "    # EDIT THESE VARIABLES FOR NEW DATASETS\n",
    "    MODEL_WEIGHTS = \"best_regression_model.pth\"\n",
    "    \n",
    "    # Example for running inference on an entirely new dataset:\n",
    "    INFERENCE_EXPERIMENT = \"test_7_simulation\"\n",
    "    INFERENCE_METADATA_PATH = f'../anr_pkg/src/processed_data/{INFERENCE_EXPERIMENT}/metadata.csv'\n",
    "    INFERENCE_PCD_DIR = f'../anr_pkg/src/processed_data/{INFERENCE_EXPERIMENT}/'\n",
    "    \n",
    "    generate_inference_csv(\n",
    "        model_path=MODEL_WEIGHTS,\n",
    "        metadata_path=INFERENCE_METADATA_PATH,\n",
    "        pcd_dir=INFERENCE_PCD_DIR,\n",
    "        device=device\n",
    "    )\n"
]

cell3_source = [
    "# ==========================================\n",
    "# MODEL RESULT ANALYTICS (Refactored)\n",
    "# ==========================================\n",
    "\n",
    "# 1. Load the inference results\n",
    "results_path = \"inference_results.csv\"\n",
    "print(f\"Loading results from {results_path}...\")\n",
    "df = pd.read_csv(results_path)\n",
    "\n",
    "# Optional: Filter for a specific workpiece if you only want to see unseen data\n",
    "# df = df[df['Filename'].str.contains(\"TH0021AV\")]\n",
    "\n",
    "y_true = df['GroundTruth_CD'].values\n",
    "y_pred = df['Predicted_CD'].values\n",
    "errors = df['Error'].values\n",
    "\n",
    "# 2. Calculate Metrics\n",
    "mae = mean_absolute_error(y_true, y_pred)\n",
    "rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
    "r2 = r2_score(y_true, y_pred)\n",
    "max_error = np.max(errors)\n",
    "\n",
    "print(f\"\\n--- PERFORMANCE METRICS ({len(df)} samples) ---\")\n",
    "print(f\"Mean Absolute Error (MAE): {mae:.4f} mm\")\n",
    "print(f\"Root Mean Squared Error (RMSE): {rmse:.4f} mm\")\n",
    "print(f\"R-Squared (R²): {r2:.4f}\")\n",
    "print(f\"Max Error: {max_error:.4f} mm\")\n",
    "print(\"-------------------------------------------\\n\")\n",
    "\n",
    "# 3. Plot Prediction vs Ground Truth\n",
    "plt.figure(figsize=(8, 8))\n",
    "plt.scatter(y_true, y_pred, alpha=0.5, color='blue', edgecolor='k')\n",
    "\n",
    "# Calculate Line of Best Fit\n",
    "m, b = np.polyfit(y_true, y_pred, 1)\n",
    "plt.plot(y_true, m*y_true + b, color='red', linestyle='--', linewidth=2, label=f'Best Fit: y = {m:.2f}x + {b:.2f}')\n",
    "\n",
    "# Perfect Prediction Line (y = x)\n",
    "min_val = min(min(y_true), min(y_pred))\n",
    "max_val = max(max(y_true), max(y_pred))\n",
    "plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='-', linewidth=1.5, label='Perfect Prediction (y=x)')\n",
    "\n",
    "plt.title('Predicted vs Ground Truth Chamfer Distance', fontsize=14)\n",
    "plt.xlabel('Ground Truth Chamfer Distance (mm)', fontsize=12)\n",
    "plt.ylabel('Predicted Chamfer Distance (mm)', fontsize=12)\n",
    "plt.legend(fontsize=12)\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# 4. Plot Error Distribution (Histogram)\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.hist(errors, bins=30, color='orange', edgecolor='black')\n",
    "plt.title('Distribution of Prediction Errors', fontsize=14)\n",
    "plt.xlabel('Absolute Error (mm)', fontsize=12)\n",
    "plt.ylabel('Frequency', fontsize=12)\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()\n"
]


new_nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell1_source
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell2_source
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell3_source
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(target_file, "w", encoding="utf-8") as f:
    json.dump(new_nb, f, indent=1)

print(f"Created {target_file} successfully!")
