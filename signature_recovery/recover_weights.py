import re
import os
import pickle
from utils import *
from collections import defaultdict
import sys
import numpy

file = open(f"{sys.argv[1]}_weight_vectors.txt" , "w") 

def intersect(left, right, nleft, nright):
    A = np.vstack((nleft, nright))
    b = np.array([np.dot(nleft, left), np.dot(nright, right)])

    # Find a particular solution
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Find the null space of A
    N = scipy.linalg.null_space(A, 1e-5)
    #print(A.shape)
    #U,S,V = np.linalg.svd(A)
    #print(U.shape, S.shape, V.shape)
    #print("V", V.shape)
    #rank = sum(1 for x in S if abs(x) > 1e-5)
    #N = V.T[:, rank:]

    
    return x0, N

# Function to generate random points on the n-2 dimensional subspace
def generate_points_on_subspace(x0, N, num_points=10):
    random_vectors = np.random.randn(N.shape[1], num_points)
    subspace_points = x0[:, np.newaxis] + N @ random_vectors
    return subspace_points.T


def vectorized_check_closest_pair_distance(points):
    # Extract the second coordinate from each point
    coords = np.array([p[1] for p in points])
    
    # Calculate pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sum(np.square(diff), axis=-1)
    
    # Set diagonal to infinity to ignore self-distances
    np.fill_diagonal(distances, -np.inf)
    
    # Find the minimum distance
    min_distance = np.max(distances)
    
    if min_distance < 1:
        return True
    else:
        return False


class CIFAR10NetPrefix(nn.Module):
    def __init__(self, layers):
        super(CIFAR10NetPrefix, self).__init__()
        if layers == 0:
            self.fcs = nn.Sequential()
        else:
            # Build layers from LAYER_SIZES: first layer is IDIM -> LAYER_SIZES[1],
            # subsequent layers follow LAYER_SIZES[i] -> LAYER_SIZES[i+1]
            fc_list = [nn.Linear(LAYER_SIZES[0], LAYER_SIZES[1])]
            for i in range(1, layers):
                fc_list.append(nn.Linear(LAYER_SIZES[i], LAYER_SIZES[i+1]))
            self.fcs = nn.Sequential(*fc_list)
        self.double()

    def relu_around(self, x):
        # Per-cell linearisation anchored on sample[0]'s pre-activation sign pattern.
        # ReLU mode (LEAKY_ALPHA == 0): mask is 0/1 — identical to original behaviour.
        # Leaky mode (LEAKY_ALPHA  > 0): mask is alpha/1 — linearises through the leak.
        if LEAKY_ALPHA > 0:
            slope = cell_slope_mask(x[:1])  # 1 on ON cells, alpha on OFF cells
            return x * slope
        mask = (x[:1] >= 0).to(torch.float64)
        return x * mask

    @torch.no_grad
    def forward_around(self, x):
        x = x.view(-1, IDIM)
        if len(self.fcs) == 0: return x
        for layer in self.fcs:
            x = self.relu_around(layer(x))
        return x

    @torch.no_grad
    def forward(self, x):
        x = x.view(-1, IDIM)
        if len(self.fcs) == 0: return x
        for layer in self.fcs:
            if LEAKY_ALPHA > 0:
                x = act(layer(x))
            else:
                x = nn.functional.relu(layer(x))
        return x

def transfer_weights(source_model, target_model, source_prefix='', target_prefix='fcs'):
    target_state_dict = {}
    source_state_dict = source_model.state_dict()
    
    layer_count = 0
    while True:
        source_weight_key = f'{source_prefix}fc{layer_count+1}.weight'
        source_bias_key = f'{source_prefix}fc{layer_count+1}.bias'
        
        if source_weight_key not in source_state_dict:
            break
        
        target_weight_key = f'{target_prefix}.{layer_count}.weight'
        target_bias_key = f'{target_prefix}.{layer_count}.bias'

        target_state_dict[target_weight_key] = source_state_dict[source_weight_key]
        target_state_dict[target_bias_key] = source_state_dict[source_bias_key]

        layer_count += 1

    target_model.load_state_dict(target_state_dict, strict=False)

    return layer_count
    
def is_consistent_help(points, prefix, layer=0, do_return_soln=False, allow_close=False):
    samples = []

    # The points need to be in different linear regions to try and compare them
    if vectorized_check_closest_pair_distance(points) and not allow_close:
        return None, None # rejected
    
    if do_return_soln:
        print("Num points", len(points))
        mid = np.stack([x[1] for x in points])
        hiddens = prefix(torch.tensor(mid).cpu()).cpu().numpy()
        hiddens = (hiddens>1e-4)
        hits = hiddens.sum(0)
        order = np.argsort(hits)
        print("Hits", hits.shape)

        # ReLU mode: zero-hit coords contribute exactly nothing to the SVD constraints,
        # so the cluster is unrecoverable (original behavior).
        # Leaky mode (LEAKY_ALPHA > 0): zero-hit coords still contribute alpha*z signal
        # via forward_around's leaky linearisation, so the SVD can still recover.
        if np.min(hits) == 0 and layer > 0:
            print("Hit some zero times. Mean OK", np.mean(hits!=0))
            print(list(hits))
            if LEAKY_ALPHA == 0:
                return None, None
            print(f"  [leaky alpha={LEAKY_ALPHA}] proceeding despite zero-hit coords (alpha*z still carries signal)")
        points_subset = []
        # hits tracks per-coord coverage in the *prefix output* (= input to the
        # target weight). Original code used LAYER_SIZES[layer+1] (output dim of
        # the target weight), which only happens to equal hiddens.shape[1] when
        # input_dim == first_hidden_dim (tiniest's uniform 8x). For tinier
        # (32->16) or any non-uniform arch, that mismatched the hiddens shape
        # and broke broadcast at hits += hiddens[entry]. hiddens.shape[1] is
        # the right size for all layers including layer 0 (prefix=identity).
        hits = np.zeros(hiddens.shape[1])

        for coord in order:
            if hits[coord] >= 4:
                continue
            for entry in np.where(hiddens[:, coord])[0][:2]:
                points_subset.append(points[entry])
                hits += hiddens[entry]
                
        points = points_subset

    for i, (left, x0, right) in enumerate(points):
        left = np.array(left)
        right = np.array(right)
        x0 = np.array(x0)

        nleft = get_normal(left)
        nright = get_normal(right)

        _, N = intersect(left, right, nleft, nright)
        points = generate_points_on_subspace(x0, N, LAYER_SIZES[layer+1]*2).tolist()

        points = np.concatenate(([x0], points), 0)
        
        points = prefix.forward_around(torch.tensor(points).cpu()).cpu()

        samples.append(points)

    samples = np.concatenate(samples, 0)

    all_zero = np.sum(np.sum(np.abs(samples),0)<1e-5)

    # We need to share at least 3 coordinates in common to try and compare
    # If we only have two there are enough free variables for anything to happen.
    shared_coords = np.sum(np.sum(np.abs(samples[::LAYER_SIZES[layer+1]*2]) > 1e-5,0) >= 2)
    #print('shared',shared_coords)
    if shared_coords <= 3:
        print("Reject")
        return None # rejected

    mean_point = np.mean(samples, axis=0)
    
    centered_samples = samples - mean_point

    if do_return_soln:
        U, S, Vt = np.linalg.svd(centered_samples)

        ans = Vt[-1]
        ans = norm(ans)
        

        return S, Vt[-1]

    tt = torch.tensor(centered_samples).double()
    S = torch.linalg.svdvals(tt).cpu().numpy()
    print(S)

    return S[len(S)-all_zero-1]

def is_consistent(points, prefix, layer=0, do_return_soln=False):
    try:
        return is_consistent_help(points, prefix, layer, do_return_soln)
    except MathIsHard:
        return None
        


def extract_weights(maybe, prefix, layer):
    if True:
        if True:
            if DEBUG:
                for i in range(len(maybe[:10])):
                    idx = cheat_neuron_diff(maybe[i][0], maybe[i][2])
                    print(i,idx,end='  ')
                print()

            print("Size", len(maybe))
            S, soln = is_consistent(maybe, prefix, layer, True)

            print('Singular values', S)
            # ReLU mode: strict gate (original). Smallest SV is the kink direction;
            # second-smallest must be substantially larger so the kink is uniquely identified.
            # Leaky mode (LEAKY_ALPHA > 0): the OFF-coord alpha*z signal adds extra small SVs,
            # so the strict gate fails. Return soln whenever SVD ran successfully — the real
            # quality check happens downstream in dosteal via min(errs) < 1e-3 against
            # the cheat solution.
            if S is not None:
                if LEAKY_ALPHA > 0:
                    return soln
                if S[-2] > 1e-2 and S[-1] < 1e-4:
                    return soln

        
def dosteal(LAYER, cluster):
    prefix = CIFAR10NetPrefix(LAYER).cpu()
    transfer_weights(cheat_net_cpu, prefix)
    layer_dir = os.path.join('/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/signature_recovery/outputs/model_weights/Vrelu', f"layer_{LAYER}")
    os.makedirs(layer_dir, exist_ok=True)
    
    for cluster_id, maybe in sorted(cluster.items(), key=lambda x: len(x[1])):
        maybe = np.array(maybe)

        if True:
            print()
            print()
            print()
            print("CLUSTER ID", cluster_id)
            print("Recovering weight given", len(maybe), "dual points")
            maybe = maybe[:1200]
            soln = extract_weights(maybe, prefix, layer)
            file.write(f'\n\nExtracted weight vector, {soln}')

            print('Extracted weight vector', soln)

            neuron_dir = os.path.join(layer_dir, f"neuron_{cluster_id}")
            os.makedirs(neuron_dir, exist_ok=True)

            # Compute error in recovering this layer
            if soln is not None:
                np_soln = np.array(soln)
                # Save raw (unscaled) recovered weights
                np.savez(os.path.join(neuron_dir, "weights_unscaled.npz"), np_soln)
                np.savetxt(os.path.join(neuron_dir, "weights_unscaled.txt"), np_soln)

                # Find best matching neuron and compute scaling factor
                errs = []
                factors = []
                for maybe_neuron in range(LAYER_SIZES[LAYER+1]):
                    factor = np.median(soln/cheat_solution[LAYER][maybe_neuron, :])
                    factors.append(factor)
                    errs.append(np.sum(np.abs(soln / factor - cheat_solution[LAYER][maybe_neuron, :])))

                if min(errs) < 1e-3:
                    best_neuron = np.argmin(errs)
                    best_error = np.min(errs)
                    best_factor = factors[best_neuron]

                    # Apply scaling factor to get correctly scaled weights
                    scaled_soln = np_soln / best_factor

                    # Save scaled weights (magnitude-corrected, unsigned)
                    np.savez(os.path.join(neuron_dir, "weights.npz"), scaled_soln)
                    np.savetxt(os.path.join(neuron_dir, "weights.txt"), scaled_soln)

                    # Save metadata
                    metadata = {
                        'matched_neuron': int(best_neuron),
                        'scaling_factor': float(best_factor),
                        'absolute_error': float(best_error),
                        'cluster_id': int(cluster_id)
                    }
                    import json
                    with open(os.path.join(neuron_dir, "metadata.json"), 'w') as mf:
                        json.dump(metadata, mf, indent=2)

                    print(f"Successfully extracted neuron {best_neuron} with abs err {best_error}")
                    print(f"  Applied scaling factor: {best_factor:.6e}")
                    file.write(f"\n\nSuccessfully extracted neuron, {best_neuron},with abs err, {best_error}")
                    file.write(f"\n  Scaling factor: {best_factor}")
                else:
                    print("Failed to identify recovered neuron")
                    file.write("\n\nFailed to identify recovered neuron")
                    # Still save unscaled as fallback
                    np.savez(os.path.join(neuron_dir, "weights.npz"), np_soln)
                    np.savetxt(os.path.join(neuron_dir, "weights.txt"), np_soln)
            else:
                print("Not enough to fully extract")
                file.write("Not enough to fully extract")
            
if __name__ == '__main__':
    layer = int(sys.argv[1])
    dosteal(layer, pickle.load(open("/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/signature_recovery/exp/1-cluster-%d.p"%layer,"rb")))
