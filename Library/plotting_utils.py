import os
import pickle
import numpy as np

def load_qgt(folder_name):
    """Load QGT entries (np object array) and meta dict from a sweep folder."""
    base = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", folder_name)
    qgt_path  = os.path.join(base, "QGT_2D.npy")
    meta_path = os.path.join(base, "meta_info.pkl")
    if not os.path.exists(qgt_path):
        raise FileNotFoundError(f"QGT data not found in '{base}'.")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    entries = np.load(qgt_path, allow_pickle=True)
    return entries, meta

def filter_entries_by_omega(entries, omega_min=None, omega_max=None):
    """Return a list of entries whose float(entry['omega']) lies in [omega_min, omega_max]."""
    def _in_range(w):
        if (omega_min is not None) and (w < omega_min): return False
        if (omega_max is not None) and (w > omega_max): return False
        return True
    filtered = [e for e in entries if _in_range(float(e["omega"]))]
    if len(filtered) == 0:
        raise ValueError("No omega slices fall within the requested range.")
    return filtered
