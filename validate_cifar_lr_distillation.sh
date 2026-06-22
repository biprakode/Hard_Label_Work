#!/usr/bin/env bash
# Validate cifar_lr extraction by running pure-distillation baseline + comparing.
#
# Preflight (REQUIRED — aborts on any failure):
#   1. X_test, X_test2, X_test3 are byte-distinct (no overlap)
#   2. Victim TinyModel_leakyrelu.pth doesn't memorise X_test3 (gap <= 1.0 pt)
#   3. cifar_lr extraction artifacts exist on disk
#
# Action:
#   4. Run run_distillation_baseline.sh (handles snapshot of extraction outputs)
#
# Output:
#   5. paper_notes/section3/reports/cifar_leakyrelu_distillation.md
#      (written by the baseline script itself — extraction-vs-distillation table)
#   6. /tmp/cifar_lr/validation.log — full preflight + baseline log

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
LOG=/tmp/cifar_lr/validation.log
mkdir -p /tmp/cifar_lr

echo "==================================================" | tee "$LOG"
echo "VALIDATE cifar_lr: $(date)"                          | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# ---------- PREFLIGHT 1 — X_test slice disjointness ----------
echo "=== [P1] X_test / X_test2 / X_test3 disjointness ===" | tee -a "$LOG"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import numpy as np, hashlib, sys
slices = {
    'X_test':  ('data/x_test.npy',        'data/y_test.npy'),
    'X_test2': ('data/x_test2_cifar.npy', 'data/y_test2_cifar.npy'),
    'X_test3': ('data/x_test3_cifar.npy', 'data/y_test3_cifar.npy'),
}
arrs, hashes = {}, {}
for name, (xp, yp) in slices.items():
    x = np.load(xp); y = np.load(yp)
    arrs[name] = (x, y)
    hashes[name] = hashlib.sha256(x.tobytes()).hexdigest()[:16]
    print(f"  {name:8s}  shape={x.shape}  sha256[:16]={hashes[name]}")

# Pairwise byte-equality test: do any two slices share the same array data?
fail = False
names = list(slices)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a = arrs[names[i]][0]
        b = arrs[names[j]][0]
        if a.shape == b.shape and np.array_equal(a, b):
            print(f"  FAIL: {names[i]} == {names[j]} byte-for-byte")
            fail = True
        if hashes[names[i]] == hashes[names[j]]:
            print(f"  FAIL: {names[i]} and {names[j]} have identical sha256")
            fail = True
        # Row-overlap test: any rows shared?
        # Hash each row, intersect
        row_a = {hashlib.md5(a[k].tobytes()).digest() for k in range(min(len(a), 10000))}
        row_b = {hashlib.md5(b[k].tobytes()).digest() for k in range(min(len(b), 10000))}
        overlap = len(row_a & row_b)
        print(f"  {names[i]} vs {names[j]}: {overlap} shared rows")
        if overlap > 0:
            print(f"  FAIL: {names[i]} and {names[j]} have {overlap} overlapping rows")
            fail = True

if fail:
    print("\nDISJOINTNESS PREFLIGHT FAILED")
    sys.exit(1)
print("\n  OK: all three X_test slices are mutually disjoint")
PY

# ---------- PREFLIGHT 2 — victim leakage diagnostic ----------
echo "=== [P2] victim leakage diagnostic (X_test3 vs X_test gap) ===" | tee -a "$LOG"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import os, numpy as np, torch, torch.nn as nn, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class FullModel(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.fc1=nn.Linear(3072,256); self.fc2=nn.Linear(256,256); self.fc3=nn.Linear(256,256)
        self.fc4=nn.Linear(256,64);   self.fc5=nn.Linear(64,10)
        self.act = nn.LeakyReLU(alpha) if alpha>0 else nn.ReLU()
    def forward(self,x):
        x=self.act(self.fc1(x)); x=self.act(self.fc2(x)); x=self.act(self.fc3(x))
        x=self.act(self.fc4(x)); return self.fc5(x)

def pre(u8): return (u8.astype(np.float64)/255.0)*2.0 - 1.0

m = FullModel(0.01).double()
sd = torch.load('tiny_stuff/TinyModel_leakyrelu.pth', map_location='cpu', weights_only=True)
m.load_state_dict(sd); m.eval()

def acc(xp, yp):
    x = pre(np.load(xp).reshape(-1, 3072)); y = np.load(yp)
    with torch.no_grad():
        p = m(torch.tensor(x)).argmax(1).numpy()
    return (p==y).mean()

a1 = acc('data/x_test.npy',        'data/y_test.npy')
a2 = acc('data/x_test2_cifar.npy', 'data/y_test2_cifar.npy')
a3 = acc('data/x_test3_cifar.npy', 'data/y_test3_cifar.npy')
gap = (a3-a1)*100
print(f"  X_test       (CIFAR test):                {a1*100:.2f}%")
print(f"  X_test2      (train[40000:50000]):        {a2*100:.2f}%")
print(f"  X_test3      (train[10000:20000]):        {a3*100:.2f}%")
print(f"  X_test3 - X_test gap = {gap:+.2f} pt   (clean if |gap| <= 1.0)")
if abs(gap) > 1.0:
    print("\nVICTIM LEAKAGE PREFLIGHT FAILED")
    sys.exit(1)
print("  OK: victim does not memorise X_test3")
PY

# ---------- PREFLIGHT 3 — extraction artifacts exist ----------
echo "=== [P3] cifar_lr extraction artifacts present ===" | tee -a "$LOG"
for f in results/reconstructed_models/reconstructed_full.pth \
         results/reconstructed_models/extraction_metrics.json; do
    if [ ! -s "$f" ]; then
        echo "  FAIL: $f missing or empty — extraction did not produce expected output" | tee -a "$LOG"
        exit 1
    fi
    echo "  OK: $f ($(du -h "$f" | cut -f1))" | tee -a "$LOG"
done

# ---------- RUN: distillation baseline ----------
echo "=== [R] run_distillation_baseline.sh ===" | tee -a "$LOG"
echo "  (this snapshots extraction outputs as *_extraction, runs pure-distillation," | tee -a "$LOG"
echo "   archives those as *_distillation, restores extraction as canonical, emits report)" | tee -a "$LOG"
PYTHON_BIN="$PY" ./run_distillation_baseline.sh 2>&1 | tee -a "$LOG"

# ---------- POST: locate the comparison report ----------
REPORT="$(cd "$HERE/../.." && pwd)/paper_notes/section3/reports/cifar_leakyrelu_distillation.md"
echo "" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "VALIDATION COMPLETE: $(date)"                       | tee -a "$LOG"
if [ -f "$REPORT" ]; then
    echo "Comparison report: $REPORT"                     | tee -a "$LOG"
    echo "---"                                            | tee -a "$LOG"
    head -40 "$REPORT"                                    | tee -a "$LOG"
else
    echo "WARNING: report not at expected path: $REPORT"  | tee -a "$LOG"
fi
echo "==================================================" | tee -a "$LOG"
