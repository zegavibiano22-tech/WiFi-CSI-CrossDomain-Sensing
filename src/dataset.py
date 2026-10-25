import os
import torch
import scipy.io as sio
import torch.nn.functional as F
from torch.utils.data import Dataset

class WidarBVPDataset(Dataset):
    """
    Dataset loader khusus untuk data Widar 3.0 BVP.
    Menyeleraskan dimensi time step secara otomatis menggunakan interpolasi.
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
        matriks = data['velocity_spectrum_ro']
        
        # Convert ke Tensor PyTorch
        tensor_bvp = torch.tensor(matriks, dtype=torch.float32)
        
        # Pastikan ada dimensi channel -> (1, H, W, T)
        if tensor_bvp.dim() == 3:
            tensor_bvp = tensor_bvp.unsqueeze(0)
            
        c, h, w, t = tensor_bvp.shape
        
        # Jika time step beda dari target (20), lakukan interpolasi
        if t != self.target_time_steps:
            # Gunakan .reshape() sebagai pengganti .view()
            tensor_bvp = tensor_bvp.reshape(c, h * w, t)
            tensor_bvp = F.interpolate(tensor_bvp, size=self.target_time_steps, mode='linear', align_corners=False)
            tensor_bvp = tensor_bvp.reshape(c, h, w, self.target_time_steps)
            
        # Permute ke format PyTorch: (Channel=1, Time=20, Height=20, Width=20)
        tensor_bvp = tensor_bvp.permute(0, 3, 1, 2).contiguous()
        
        # Parse label gesture (Y) & environment/domain (D)
        parts = filename.split('-')
        gesture_label = int(parts[1]) - 1     # Target gesture
        env_label = int(parts[2]) - 1         # Target domain
        
        return (
            tensor_bvp, 
            torch.tensor(gesture_label, dtype=torch.long), 
            torch.tensor(env_label, dtype=torch.long)
        )
