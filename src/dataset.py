import os
import re
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

class WidarBVPDataset(Dataset):
    """
    Dataset loader khusus untuk data Widar 3.0 BVP (Body Velocity Profile).
    Mengekstrak fitur BVP, Gesture Label (Y), dan Domain/Environment Label (D).
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        filename = os.path.basename(file_path)
        
        # Load matriks BVP
        data = sio.loadmat(file_path)
        matriks = data['velocity_spectrum_ro']
        
        # Parse label gesture (Y) & environment/domain (D)
        parts = filename.split('-')
        gesture_label = int(parts[1]) - 1     # Target gesture/kelas
        env_label = int(parts[2]) - 1         # Target domain
        
        # Format Tensor PyTorch: (Channel=1, Time=20, Height=20, Width=20)
        tensor_bvp = torch.tensor(matriks, dtype=torch.float32).unsqueeze(0)
        
        return (
            tensor_bvp, 
            torch.tensor(gesture_label, dtype=torch.long), 
            torch.tensor(env_label, dtype=torch.long)
        )
