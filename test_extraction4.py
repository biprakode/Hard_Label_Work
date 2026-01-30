import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from PIL import Image
import os

# --- 0. Setup Paths (Based on user's last input) ---
CSV_FILE_DIR = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/tu results tiny/with_extracted"
TRUE_MODEL_PATH = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/tiny_shit/TinyModel_relu.pth" 

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 64) 
        self.l1 = nn.Linear(64, 64) 
        self.l2 = nn.Linear(64, 64) 
        self.l3 = nn.Linear(64, 64) # <-- The missing hidden layer (fc4)
        self.l4 = nn.Linear(64, 10) # <-- The final output layer

    def forward(self, x):
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x)) # Activation for the 4th hidden layer
        x = self.l4(x)
        return x

# --- 2. Corrected CSV Load Function (Same parsing logic) ---
def load_layer(csv_path, out_dim, in_dim, column_name):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        W = np.zeros((out_dim, in_dim), dtype=np.float32)
        return torch.tensor(W, dtype=torch.float32)

    extracted = []
    df.columns = df.columns.str.strip()
    
    for _, row in df.iterrows():
        neuron_id = int(row["neuronID"])
        extracted_data_str = str(row[column_name])
        cleaned_str = extracted_data_str.replace('\n', ' ').replace('[', ' ').replace(']', ' ').strip(' "')
        w = np.fromstring(cleaned_str, sep=' ')
        if w.shape[0] != in_dim:
            w = np.zeros(in_dim, dtype=np.float32)
        extracted.append((neuron_id, w))

    W = np.zeros((out_dim, in_dim), dtype=np.float32)
    for neuron_id, w in extracted:
        if w.shape[0] == in_dim and neuron_id < out_dim:
            W[neuron_id] = w

    return torch.tensor(W, dtype=torch.float32)


def load_extracted_model(model, file_dir, true_model_path):
    # Layer 0-3: Load from CSV files
    csv_layer_paths = [
        ("layer_0_comparison.csv", model.l0, 64, 64),
        ("layer_1_comparison.csv", model.l1, 64, 64),
        ("layer_2_comparison.csv", model.l2, 64, 64),
        ("layer_3_comparison.csv", model.l3, 64, 64) # Now 64x64, as expected
    ]
    
    print("Loading 4 hidden layers from CSV...")
    for filename, layer, out_dim, in_dim in csv_layer_paths:
        full_path = os.path.join(file_dir, filename)
        weight_data = load_layer(full_path, out_dim, in_dim, column_name="extracted")
        layer.weight.data = weight_data

    print("Loading final output layer (l4) from .pth file...")
    try:
        state_dict = torch.load(true_model_path)
        
        # Check for bias/weight keys specific to the output layer in the PTH file
        if "output.weight" in state_dict and "output.bias" in state_dict:
            model.l4.weight.data = state_dict["output.weight"]
            model.l4.bias.data = state_dict["output.bias"]
        else:
            print("WARNING: 'output.weight' not found in .pth file. l4 layer initialized with random weights.")
            
    except Exception as e:
        print(f"ERROR: Could not load L4 from .pth file: {e}")

    return model

def load_true_model(model, true_model_path):
    print("Loading all 5 layers from .pth file with correct remapping...")
    state_dict = torch.load(true_model_path)
    
    # Correct key mapping for the 5-layer architecture
    key_mapping = {
        "fc1.weight": "l0.weight", "fc1.bias": "l0.bias",
        "fc2.weight": "l1.weight", "fc2.bias": "l1.bias",
        "fc3.weight": "l2.weight", "fc3.bias": "l2.bias",
        "fc4.weight": "l3.weight", "fc4.bias": "l3.bias",    # <-- Maps fc4 to l3
        "output.weight": "l4.weight", "output.bias": "l4.bias", # <-- Maps output to l4
    }
    
    new_state_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in state_dict:
            new_state_dict[new_key] = state_dict[old_key]
        else:
            # Handle the case where a key might be missing (e.g., if bias was disabled)
            pass

    model.load_state_dict(new_state_dict)
    return model

def preprocess_images(images):
    images = tf.image.resize(images, (8, 8))
    images = tf.image.rgb_to_grayscale(images)
    images = tf.squeeze(images, axis=-1)
    return images.numpy()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_test_resized = preprocess_images(x_test)
x_test_resized = x_test_resized.astype("float32") / 255.0
x_test_flat = x_test_resized.reshape(-1, 64)

X_test = torch.tensor(x_test_flat, dtype=torch.float32)
Y_test = torch.tensor(y_test.squeeze(), dtype=torch.long)

def test_accuracy(model, X_test, Y_test, model_name):
    model.eval()
    correct = (model(X_test).argmax(dim=1) == Y_test).sum().item()
    acc = correct / len(Y_test)
    print(f"\n--- Model Test Result ---")
    print(f" {model_name} Accuracy: {acc:.4f}")
    return acc



# 4A. Extracted Weights Model (CSV for L0-L3, PTH for L4)
print("\nLoading Extracted Model...")
model_extracted = TinyModel()
model_extracted = load_extracted_model(model_extracted, CSV_FILE_DIR, TRUE_MODEL_PATH) 
test_accuracy(model_extracted, X_test, Y_test, "Extracted Model (CSV L0-L3 + PTH L4)")


print("\nLoading True Model...")
model_true = TinyModel()
try:
    model_true = load_true_model(model_true, TRUE_MODEL_PATH)
    print(f"Successfully loaded true weights from {TRUE_MODEL_PATH} after 5-layer remapping.")
    test_accuracy(model_true, X_test, Y_test, "True Model (All 5 layers from PTH)")

except Exception as e:
    print(f"CRITICAL ERROR loading true model: {e}")