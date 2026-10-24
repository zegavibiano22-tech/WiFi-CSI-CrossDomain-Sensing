import os, re, glob
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader

TARGET_T = 20

def scan_dataset(data_root):
    """Scan data_root/userX/*.mat -> DataFrame (user, gesture, torso, face, rep, rx)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(data_root, "**", "*.mat"), recursive=True)):
        name = os.path.basename(path)
        p = name.split("-")
        try:
            user = int(re.search(r"\d+", p[0]).group())
            g, tq, fc, rep, rx = (int(p[i]) for i in range(1, 6))
        except (IndexError, ValueError, AttributeError):
            print("Skip (format nama tidak dikenali):", name)
            continue
        rows.append(dict(path=path, user=user, gesture=g, torso=tq, face=fc, rep=rep, rx=rx,
                         inst=f"{user}-{g}-{tq}-{fc}-{rep}"))   # 1 'inst' = 1 eksekusi gestur
    return pd.DataFrame(rows)


def select_samples(df, gestures=(1, 2, 3, 4, 5, 6), torso=None, face=None,
                   max_per_group=None, seed=42):
    d = df[df.gesture.isin(gestures)]
    if torso is not None:
        d = d[d.torso.isin(torso)]
    if face is not None:
        d = d[d.face.isin(face)]
    if max_per_group:
        d = d.sample(frac=1, random_state=seed).groupby(["user", "gesture"]).head(max_per_group)
    return d.reset_index(drop=True)


def split_louo(df, target_user, val_ratio=0.2, seed=42):
    """Leave-one-user-out. Val dipisah per 'inst' agar file satu eksekusi gestur
    (receiver berbeda) tidak bocor antara train dan val."""
    src = df[df.user != target_user]
    tgt = df[df.user == target_user]
    rng = np.random.RandomState(seed)
    inst = src["inst"].unique()
    val_inst = rng.choice(inst, size=max(1, int(len(inst) * val_ratio)), replace=False)
    is_val = src["inst"].isin(val_inst)
    return (src[~is_val].reset_index(drop=True),
            src[is_val].reset_index(drop=True),
            tgt.reset_index(drop=True))


def temporal_lowpass(x, wn=0.4, order=2):
    """Butterworth low-pass sepanjang sumbu waktu. x: (T,H,W). wn relatif terhadap Nyquist."""
    b, a = butter(order, wn)
    return filtfilt(b, a, x, axis=0)


def load_bvp(path, T=TARGET_T, smooth_wn=None):
    """BVP (H,W,T0) -> tensor (1,T,H,W): resample waktu ke T, (opsional) LPF, normalisasi max."""
    m = sio.loadmat(path)["velocity_spectrum_ro"].astype(np.float32)
    h, w, t0 = m.shape
    x = torch.from_numpy(m).reshape(h * w, t0).unsqueeze(0)
    if t0 != T:
        x = F.interpolate(x, size=T, mode="linear", align_corners=False)
    x = x.squeeze(0).reshape(h, w, T).permute(2, 0, 1).contiguous()       # (T,H,W)
    if smooth_wn is not None:
        x = torch.from_numpy(temporal_lowpass(x.numpy(), smooth_wn).copy()).float().clamp(min=0)
    x = x / (x.max() + 1e-8)
    return x.unsqueeze(0)


class BVPDataset(Dataset):
    def __init__(self, df, smooth_wn=None):
        self.df = df.reset_index(drop=True)
        self.x = torch.stack([load_bvp(p, smooth_wn=smooth_wn) for p in self.df.path])
        self.y = torch.tensor(self.df.gesture.values - 1, dtype=torch.long)   # 0..5
        self.u = torch.tensor(self.df.user.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.u[i]


def make_fold(df, target_user, batch_size=16, val_ratio=0.2, seed=42, smooth_wn=None):
    tr, va, te = split_louo(df, target_user, val_ratio, seed)
    ds_tr, ds_va, ds_te = (BVPDataset(d, smooth_wn) for d in (tr, va, te))
    kw = dict(batch_size=batch_size, num_workers=0)
    return dict(
        train=DataLoader(ds_tr, shuffle=True, drop_last=True, **kw),
        val=DataLoader(ds_va, shuffle=False, **kw),
        target=DataLoader(ds_te, shuffle=False, **kw),                           # evaluasi (pakai label)
        target_unlabeled=DataLoader(ds_te, shuffle=True, drop_last=True, **kw),  # DANN (label TIDAK dipakai)
        sizes=(len(ds_tr), len(ds_va), len(ds_te)),
    )
