import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MotionData(Dataset):
    def __init__(self, label_file, data_dir, transform=None):
        self.labels = pd.read_csv(label_file)
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename = self.labels.iloc[idx]['filename']
        filename = os.path.join(self.data_dir, (filename + ".csv"))
        df = pd.read_csv(filename)
        data = df[['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z']].values.astype(np.float32)

        # normalize
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        data = (data - min_val) / (max_val - min_val)

        if self.transform:
            data = self.transform(data)

        label = self.labels.iloc[idx]['0']

        return data, label