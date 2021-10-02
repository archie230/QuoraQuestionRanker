import numpy as np
import math

# NDCG
def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return y_value
    if gain_scheme == 'exp2':
        return 2**y_value - 1
    
    raise ValueError()

def dcg(ys_true: np.array, ys_pred: np.array, gain_scheme: str, k: int = None) -> float:
    indices = np.argsort(ys_pred)[::-1]
    disc_gains = np.array(
        [
            compute_gain(j, gain_scheme) / math.log2(i+2)\
            for i, j in enumerate(ys_true[indices].astype('float64'))
        ], dtype=np.float64
    )
    
    return disc_gains[:k].sum()


def compute_ideal_dcg(ys_true: np.array, gain_scheme: str, k: int = None):
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme, k)
    
    return ideal_dcg


def ndcg(ys_true: np.array, ys_pred: np.array, gain_scheme: str = 'const', k: int = None) -> float:
    dcg_val = dcg(ys_true, ys_pred, gain_scheme, k)
    ideal_dcg = compute_ideal_dcg(ys_true, gain_scheme, k)
    
    return dcg_val / ideal_dcg