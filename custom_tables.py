import pandas as pd
import numpy as np
import os
import sys
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def get_alpha(votes_p, votes_m, n): 
    if n == 0:
        return 0.9999999999999999999999999999999999999999
    p_observed = max(votes_m, votes_p) / n
    epsilon = p_observed - 0.5
    alpha = np.exp(-2 * epsilon**2 * n)
    return alpha


def analyze_df(df):
    df["votes+"] = df["Vote dOFF>dON"].cumsum()     # correct sign
    df["votes-"] = (~df["Vote dOFF>dON"]).cumsum()  # incorrect sign
    return {
        "votes+": df["votes+"].values[-1],
        "votes-": df["votes-"].values[-1],
        "vON/vOFF" : df[""] 
    }


if __name__ == "__main__":
    all_rows = []   # this will hold ALL neurons from ALL layers
    for layerID in range(1, 5):
        N_NEURONS = 64 if layerID == 4 else 256
        for neuronID in range(N_NEURONS):

            path = f"/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/sign_recovery/results/tu_tiny_relu/sign_results/results/model_TinyModel_relu/layerID_{layerID}/neuronID_{neuronID}/df.pkl"
            if not os.path.exists(path):
                continue
            df = pd.read_pickle(path)
            # get votes only
            res = analyze_df(df)
            res["layerID"] = layerID
            res["neuronID"] = neuronID
            all_rows.append(res)

    out = pd.DataFrame(all_rows)
    out = out[["layerID", "neuronID", "votes+", "votes-"]]

    out.to_csv("/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/sign_recovery/results/tu_tiny_relu/sign_recovery_votes.csv", index=False)
    print("Saved: sign_recovery_votes.csv")
