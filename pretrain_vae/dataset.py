import torch
import numpy as np

class VOGMaze2dOfflineRLDataset(torch.utils.data.Dataset):
    '''
    Offline RL dataset for 2D maze environments from OG-Bench.
    - Dataset: pointmaze-giant-navigate-v0
        - Total samples: 1000500, Subtrajectory length: 2001
        - Observation shape: (1000500, 64, 64, 3)
        - Observation mean:  141.785487953533
        - Observation std:  71.0288272312382
        - Action shape: (1000500, 2)
        - Action mean: [-0.00524961 -0.00168911]
        - Action std:  [0.70124096 0.6971626]
    '''
        
    def __init__(self, dataset_url='./dataset/visual-pointmaze-medium-navigate-v0.npz' , split: str = "training"):
        
        super().__init__()
        self.dataset_url = dataset_url
        self.split = split
        dataset = self.get_dataset(self.dataset_url)
        self.observations = dataset["observations"]
        self.pos = dataset["qpos"]

        # Normalizations
        self.obs_mean = 141.785487953533
        self.obs_std = 71.0288272312382
        self.observations = (self.observations - self.obs_mean) / self.obs_std

    def __getitem__(self, idx):
        observation = torch.from_numpy(self.observations[idx]).float().permute(2, 0, 1)  # 변환 # (episode_len, obs_dim)
        pos = torch.from_numpy(self.pos[idx]).float()
        return observation, pos
    
    def __len__(self):
        return len(self.observations)

    def get_dataset(self, path):
        if self.split == "validation":
            path = path.replace(".npz", "-val.npz")
        dataset = np.load(path, allow_pickle=True, mmap_mode='r')  # 메모리 매핑 적용
        return dataset
    