# Cell 0
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import importlib
import open3d as o3d

# ==========================================
# 1. PATH & DEVICE CONFIGURATION
# ==========================================
BASE_DIR = os.getcwd()
sys.path.append(os.path.join(BASE_DIR, 'models'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Project Root: {BASE_DIR}")
print(f"Using Device: {device}")

# ==========================================
# 2. REGRESSION DATASET CLASS
# ==========================================
class ChamferRegressionDataset(Dataset):
    def __init__(self, metadata, pcd_dir, indices, num_points=1024, is_training=False, y_mean=0.0, y_std=1.0, use_norm=True):
        self.metadata = metadata.iloc[indices].reset_index(drop=True)
        self.pcd_dir = pcd_dir
        self.num_points = num_points
        self.is_training = is_training
        self.cache = {}
        self.y_mean = y_mean
        self.y_std = y_std
        self.use_norm = use_norm

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if idx in self.cache:
            points, chamfer_out = self.cache[idx]
        else:
            filename = self.metadata.iloc[idx, 0]
            chamfer = self.metadata.iloc[idx, 3] # Keep raw value

            file_path = os.path.join(self.pcd_dir, filename)
            pcd = o3d.io.read_point_cloud(file_path)

            # FPS sampling
            if len(pcd.points) > self.num_points:
                pcd = pcd.farthest_point_down_sample(self.num_points)

            # Normals
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # pcd.orient_normals_towards_camera_location([0, 0, 0])

            xyz = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            points = np.hstack((xyz, normals))

            # Spatial Normalization (Keep this: PointNet needs -1 to 1 input)
            points[:, 0:3] -= np.mean(points[:, 0:3], axis=0)
            scale = np.max(np.linalg.norm(points[:, 0:3], axis=1))
            if scale > 0:
                pass # REMOVED: points[:, 0:3] /= scale to preserve absolute scale geometry

            # Padding if necessary
            if len(points) < self.num_points:
                idx_sample = np.random.choice(len(points), self.num_points, replace=True)
                points = points[idx_sample]

            # Normalize target chamfer
            chamfer_out = (chamfer - self.y_mean) / self.y_std if self.use_norm else chamfer
            self.cache[idx] = (points, chamfer_out)

        points_np = points.copy()
        if self.is_training:
            import provider
            points_np = np.expand_dims(points_np, axis=0)
            # Perturb on X, Y, Z simultaneously up to 5 degrees (0.08726 radians)
            points_np = provider.rotate_perturbation_point_cloud_with_normal(points_np, angle_sigma=0.06, angle_clip=0.08726)
            points_np = points_np.squeeze(0)

        points_tensor = torch.tensor(points_np, dtype=torch.float32).transpose(1, 0)
        # Target is now raw units
        target_tensor = torch.tensor([chamfer_out], dtype=torch.float32)

        return points_tensor, target_tensor
    
# ==========================================
# 3. TRAINING LOOP
# ==========================================
def run_training(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, y_std=1.0, use_norm=True):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for points, target in train_loader:
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            pred, trans_feat, _ = model(points)
            
            loss = criterion(pred, target, trans_feat)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
            # Log learning rate occasionally
            if epoch % 50 == 0:
                print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        avg_train_loss = train_loss / len(train_loader)
        if use_norm:
            avg_train_loss *= (y_std ** 2)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for points, target in val_loader:
                points, target = points.to(device), target.to(device)
                pred, trans_feat, _ = model(points)
                v_loss = criterion(pred, target, trans_feat)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        if use_norm:
            avg_val_loss *= (y_std ** 2)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {avg_train_loss:.2f} | Val Loss (MSE): {avg_val_loss:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_regression_model.pth')
            print(f">>> New Best Model Saved: {avg_val_loss:.4f}")

    return history

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 1000
    USE_NORMALIZATION = False 

    METADATA_PATH = '../anr_pkg/src/processed_data/training_second/metadata.csv'
    PCD_DATA_DIR = '../anr_pkg/src/viewpoints_candidate/testing_data/training_second/'

    metadata = pd.read_csv(METADATA_PATH)

    indices = np.arange(len(metadata))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # calculate target stats
    y_mean = metadata.iloc[train_indices, 3].mean()
    y_std = metadata.iloc[train_indices, 3].std()
    print(f"Target Mean: {y_mean:.4f} | Target Std: {y_std:.4f}")

    # datasets
    train_ds = ChamferRegressionDataset(metadata, PCD_DATA_DIR, train_indices, is_training=True, y_mean=y_mean, y_std=y_std, use_norm=USE_NORMALIZATION)
    val_ds   = ChamferRegressionDataset(metadata, PCD_DATA_DIR, val_indices, is_training=False, y_mean=y_mean, y_std=y_std, use_norm=USE_NORMALIZATION)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    reg_module = importlib.import_module('pointnet_regression')
    model = reg_module.get_model(normal_channel=True).to(device)
    criterion = reg_module.get_loss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = None

    print(f"Training on: {len(train_ds)} | Validation: {len(val_ds)}")

    train_history = run_training(model, train_loader, val_loader, optimizer, scheduler, criterion, EPOCHS, y_std=y_std, use_norm=USE_NORMALIZATION)

# Cell 1
import matplotlib.pyplot as plt

def plot_learning_curves(history):
    plt.figure(figsize=(10, 6))
    
    # Plot Training Loss
    plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    
    # Plot Validation Loss
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linestyle='--', linewidth=2)
    
    plt.title('PointNet Regression Training Trend', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.ylim(0, 100)
    # plt.xlim(0, 500)
    
    # Save the plot as an image for your thesis report
    plt.savefig('training_trend.png')
    plt.show()

# Generate the plot
plot_learning_curves(train_history)

# Cell 2
# Inference

import torch
import importlib

INDEX = 8  # Change this index to test different samples from the dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

reg_module = importlib.import_module('pointnet_regression')
model = reg_module.get_model(normal_channel=True).to(device)

model.load_state_dict(torch.load('./best_regression_model.pth'))
model.eval()

points, target = val_ds[INDEX]   # reuse dataset pipeline

points = points.unsqueeze(0).to(device)   # [1, 6, 1024]

with torch.no_grad():
    pred, __, __ = model(points)


pred_norm = pred.squeeze(0).cpu().numpy()[0]

USE_NORMALIZATION = False
if USE_NORMALIZATION:
    pred_final = pred_norm * y_std + y_mean
    target_final = target.numpy()[0] * y_std + y_mean
else:
    pred_final = pred_norm
    target_final = target.numpy()[0]

print("Predicted CD (Original Units): ", pred_final)
print("Ground Truth CD (Original Units): ", target_final)

# Cell 3
# Make 'prediction_results.csv' code here


import pandas as pd
import torch
import importlib

def generate_prediction_csv(model_path, dataset, output_file='prediction_results.csv', use_norm=True, y_mean=0.0, y_std=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model architecture and weights
    reg_module = importlib.import_module('pointnet_regression')
    model = reg_module.get_model(normal_channel=True).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    results = []
    
    print(f"Evaluating {len(dataset)} samples...\n")
    with torch.no_grad():
        for idx in range(len(dataset)):
            points, target = dataset[idx]
            points = points.unsqueeze(0).to(device)
            
            pred, _, _ = model(points)
            pred_norm = pred.squeeze(0).cpu().numpy()[0]
            target_val = target.numpy()[0]
            
            if use_norm:
                pred_final = pred_norm * y_std + y_mean
                target_final = target_val * y_std + y_mean
            else:
                pred_final = pred_norm
                target_final = target_val
                
            error = abs(pred_final - target_final)
            
            results.append({
                'Index': idx,
                'GroundTruth_CD': target_final,
                'Predicted_CD': pred_final,
                'Absolute_Error': error
            })
            
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Calculate and print summary statistics
    print("Summary Statistics:")
    print(f"GroundTruth_CD avg\t{df['GroundTruth_CD'].mean():.9f}")
    print(f"GroundTruth_max\t{df['GroundTruth_CD'].max():.9f}")
    print(f"GroundTruth_min\t{df['GroundTruth_CD'].min():.9f}")
    print(f"GroundTruth_stdev\t{df['GroundTruth_CD'].std():.9f}\n")
    
    print(f"predicted_CD avg\t{df['Predicted_CD'].mean():.9f}")
    print(f"predicted_max\t{df['Predicted_CD'].max():.9f}")
    print(f"predicted_min\t{df['Predicted_CD'].min():.9f}")
    print(f"predicted_stdev\t{df['Predicted_CD'].std():.9f}\n")
    
    print(f"Error avg\t{df['Absolute_Error'].mean():.9f}")
    print(f"\nSaved to {output_file}")

# Example usage on the validation dataset:
generate_prediction_csv('./best_regression_model.pth', val_ds, output_file='predictions_results.csv', use_norm=USE_NORMALIZATION, y_mean=y_mean, y_std=y_std)


# Cell 4
# Check Processed PCD

import open3d as o3d
import numpy as np

# 1. Get the sample from your dataset
# sample_pts is [6, 1024]
sample_pts, sample_targ = train_ds[INDEX]

# 2. Convert to NumPy and Transpose back to [1024, 6]
points_np = sample_pts.numpy().T 

# 3. Separate XYZ and Normals
xyz = points_np[:, :3]
normals = points_np[:, 3:]

# 4. Create Open3D object
pcd_to_vis = o3d.geometry.PointCloud()
pcd_to_vis.points = o3d.utility.Vector3dVector(xyz)
pcd_to_vis.normals = o3d.utility.Vector3dVector(normals)

print(f"Visualizing Sample with Targets: {sample_targ.tolist()}")
print("Close the Open3D window to continue...")

# 5. Draw
o3d.visualization.draw_geometries([pcd_to_vis], 
                                  window_name="Fixed Sampled PCD (1024 points)",
                                  width=800, height=600)

# Cell 5
# INFERENCE CSV #

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

def generate_inference_csv(model, train_indices, val_indices, metadata, pcd_dir, device, y_mean, y_std, use_norm):
    model.eval()
    results = []

    # Helper to process a split
    def process_split(indices, split_name):
        # Create a dataset for inference (no training augmentations)
        ds = ChamferRegressionDataset(
            metadata, pcd_dir, indices, 
            is_training=False, 
            y_mean=y_mean, y_std=y_std, 
            use_norm=use_norm
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for i, (points, target) in enumerate(loader):
                points = points.to(device)
                pred, _, _ = model(points)
                
                # Convert back to raw units if normalization was used during training
                if use_norm:
                    pred_val = pred.item() * y_std + y_mean
                    gt_val = target.item() * y_std + y_mean
                else:
                    pred_val = pred.item()
                    gt_val = target.item()

                error = abs(gt_val - pred_val)
                
                results.append({
                    'Split': split_name,
                    'GroundTruth_CD': gt_val,
                    'Predicted_CD': pred_val,
                    'Error': error
                })

    # Process both sets
    print("Inference: Processing Training Split...")
    process_split(train_indices, 'Training')
    print("Inference: Processing Validation Split...")
    process_split(val_indices, 'Validation')

    # Create DataFrame and Save
    df_results = pd.DataFrame(results)
    # Sort by GroundTruth_CD descending to match your example format
    df_results = df_results.sort_values(by='GroundTruth_CD', ascending=False)
    
    save_path = 'inference_results.csv'
    df_results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

# ==========================================
# EXECUTION (Add this to your __main__ block)
# ==========================================
if __name__ == "__main__":
    # ... [Your existing training code] ...

    # Load the best weights
    model.load_state_dict(torch.load('best_regression_model.pth'))
    
    # Generate the CSV
    generate_inference_csv(
        model, 
        train_indices, 
        val_indices, 
        metadata, 
        PCD_DATA_DIR, 
        device, 
        y_mean, 
        y_std, 
        USE_NORMALIZATION
    )

# Cell 6
# open csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = f"../anr_pkg/src/processed_data/training_second/metadata.csv"
df = pd.read_csv(csv_path)

# Define your bins and plot
plt.hist(df['chamfer_value'], bins=30, edgecolor='black')

# Set ticks from 0 to the maximum value, step by 2 (change 2 to 0.5 or 1 as needed)
max_val = df['chamfer_value'].max()
plt.xticks(np.arange(0, max_val + 2, 2)) 

plt.xlabel('Chamfer Value')
plt.ylabel('Frequency')
plt.xlim(4,48)
plt.title('Distribution of Chamfer Values')
plt.grid(True, alpha=0.3)
plt.show()

# Cell 7
# MODEL RESULT ANALYTICS #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('./milestone1.csv')

# 1. Parity Plot
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(df['GroundTruth'], df['Predicted'], alpha=0.5, color='blue')
max_val = max(df['GroundTruth'].max(), df['Predicted'].max())
ax1.plot([0, max_val], [0, max_val], 'r--', label='Ideal (y=x)')
ax1.set_xlabel('Ground Truth Chamfer Distance (mm)')
ax1.set_ylabel('Predicted Chamfer Distance (mm)')
ax1.set_title('Parity Plot: Predicted vs. Actual Performance')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)
plt.savefig('parity_plot.png')

# 2. Viewpoint Error Heatmap
# Pivoting the data for the heatmap
pivot_df = df.pivot_table(index='elevation', columns='azimuth', values='Error(mm)', aggfunc='mean')
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot_df, annot=False, cmap='YlOrRd', ax=ax2)
ax2.set_title('Viewpoint Error Heatmap (Elevation vs. Azimuth)')
ax2.set_xlabel('Azimuth (Degrees)')
ax2.set_ylabel('Elevation (Degrees)')
plt.savefig('error_heatmap.png')

# 3. Polar Spider Plot
# Convert azimuth to radians for polar plot
df_sorted = df.sort_values('azimuth')
# Grouping by azimuth to get a cleaner spider web
azimuth_means = df_sorted.groupby('azimuth')['GroundTruth'].mean().reset_index()
angles = np.deg2rad(azimuth_means['azimuth'])
values = azimuth_means['GroundTruth']

# To close the circle
angles = np.append(angles, angles[0])
values = np.append(values, values[0])

fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax3.plot(angles, values, linewidth=2, linestyle='solid', color='green')
ax3.fill(angles, values, color='green', alpha=0.1)
ax3.set_title('Polar Representation of Ground Truth CD by Azimuth', va='bottom')
plt.savefig('polar_spider_plot.png')