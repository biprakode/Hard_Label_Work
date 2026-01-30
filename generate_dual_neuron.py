import pickle
import numpy as np
import os

cluster_files = [
    "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/signature_recovery/exp/1-cluster-0.p",
    "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/signature_recovery/exp/1-cluster-1.p",
    "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/signature_recovery/exp/1-cluster-2.p",
    "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/signature_recovery/exp/1-cluster-3.p",
    "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/signature_recovery/exp/1-cluster-4.p"
]
output_dir = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/sign_recovery/layer_neuron_npys"
os.makedirs(output_dir, exist_ok=True)

layer_boundaries = [0, 64, 128, 192, 256, 320]

def get_layer_index(neuron_idx):
    for i in range(len(layer_boundaries) - 1):
        if layer_boundaries[i] <= neuron_idx < layer_boundaries[i + 1]:
            return i
    raise ValueError(f"Invalid neuron index: {neuron_idx}")

def process_cluster_file(cluster_path):
    with open(cluster_path, "rb") as f:
        cluster_dict = pickle.load(f)
    
    data = {}
    for neuron_idx, dual_triplets in cluster_dict.items():
        if isinstance(dual_triplets, (list, tuple)):
            middle_duals = [triplet[1] for triplet in dual_triplets if len(triplet) == 3]
            dual_array = np.array(middle_duals, dtype=np.float32)
            dual_array = np.abs(dual_array) # abs weights
        else:
            raise ValueError(f"Unexpected structure for neuron {neuron_idx}")
        
        data[neuron_idx] = dual_array
    return data

all_neuron_data = {}
for cluster_file in cluster_files:
    cluster_data = process_cluster_file(cluster_file)
    all_neuron_data.update(cluster_data)

for neuron_idx, dual_array in all_neuron_data.items():
    layer_idx = get_layer_index(neuron_idx)
    file_name = f"layer{layer_idx+1}_neuron{neuron_idx%64}.npy"
    np.save(os.path.join(output_dir, file_name), dual_array)

print(f"Generated {len(all_neuron_data)} .npy files in '{output_dir}'")
