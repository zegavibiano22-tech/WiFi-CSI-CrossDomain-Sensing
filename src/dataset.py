import os
import torch
import scipy.io as sio
import torch.nn.functional as F
from torch.utils.data import Dataset

class WidarBVPDataset(Dataset):
    """
    Dataset loader khusus untuk data Widar 3.0 BVP.
    Mengubah ukuran dimensi waktu secara otomatis agar seragam (misal: 20 frame).
    """
    def __init__(self, file_paths, target_time_steps=20):
        self.file_paths = file_paths
        self.target_time_steps = target_time_steps

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        filename = os.path.basename(file_path)
        
        # Load matriks BVP
        data = sio.loadmat(file_path)
        matriks = data['velocity_spectrum_ro'] # Misal shape: (20, 20, T)
        
        # Convert ke Tensor PyTorch
        tensor_bvp = torch.tensor(matriks, dtype=torch.float32)
        
        # Atur dimensi agar berbentuk (1, H, W, T)
        if tensor_bvp.dim() == 3:
            tensor_bvp = tensor_bvp.unsqueeze(0)
            
        # Samakan ukuran dimensi waktu (T) menjadi target_time_steps (20) menggunakan interpolasi
        curr_t = tensor_bvp.shape[-1]
        if curr_t != self.target_time_steps:
            # Reshape sementara untuk interpolasi 1D pada dimensi waktu
            # (1, 1*20*20, T) -> (1, 400, T)
            c, h, w, t = tensor_bvp.shape
            tensor_bvp = tensor_bvp.view(c, h * w, t)
            
            # Interpolasi ke target_time_steps
            tensor_bvp = F.interpolate(tensor_bvp, size=self.target_time_steps, mode='linear', align_corners=False)
            
            # Kembalikan ke shape (1, T, H, W)
            tensor_bvp = tensor_bvp.view(c, h, w, self.target_time_steps)
            
        # Permute agar format akhirnya (Channel=1, Time=20, Height=20, Width=20)
        tensor_bvp = tensor_bvp.permute(0, 3, 1, 2)
        
        # Parse label gesture (Y) & environment/domain (D)
        parts = filename.split('-')
        gesture_label = int(parts[1]) - 1     # Target gesture
        env_label = int(parts[2]) - 1         # Target domain
        
        return (
            tensor_bvp, 
            torch.tensor(gesture_label, dtype=torch.long), 
            torch.tensor(env_label, dtype=torch.long)
        )
