import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EyeGaze360Dataset(Dataset):
    def __init__(self, pkl_path, phase='train', max_length=18):
        self.phase = phase
        self.max_length = max_length

        # Load the dataset
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)

        # self.data[self.phase] is a dict: {image_name: { ... }}
        phase_data = self.data[self.phase]

        # Create a list of (image_name, scanpath_idx) pairs
        self.samples = []
        for img_name, img_dict in phase_data.items():
            n_scanpath = img_dict['scanpaths'].shape[0]
            for sp_idx in range(n_scanpath):
                self.samples.append((img_name, sp_idx))

        # Optionally, store info if needed
        self.info = self.data['info'][self.phase]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name, sp_idx = self.samples[index]

        # Retrieve image-level data
        img_data = self.data[self.phase][img_name]
        image = img_data['image']  # Tensor[3, 128, 256]
        scanpaths = img_data['scanpaths']  # Tensor[n_scanpath, n_gaze_point, 3]
        frameindex = img_data['frameindex']  # np.array [n_scanpath, n_gaze_point]
        csv_ids = img_data['csv_ids']        # list[str] length n_scanpath
        image_feats = img_data['image_feats'] # (n_scanpath, n_gaze_point, 2048)
        image_labels = img_data['image_labels'] # (n_scanpath, n_gaze_point)
        sem_feats = img_data['sem_feats']    # (n_scanpath, n_gaze_point, 512)

        # Select the specific scanpath
        sp = scanpaths[sp_idx]      # shape: (n_gaze_point, 3)
        sp_image_feats = image_feats[sp_idx] # (n_gaze_point, 2048)
        sp_sem_feats = sem_feats[sp_idx]     # (n_gaze_point, 512)
        sp_csv_id = csv_ids[sp_idx]          # str representing that testee
        sp_length = sp.shape[0]

        # Truncate or pad to max_length
        valid_len = min(sp_length, self.max_length)
        # scanpath
        scanpath = torch.zeros(self.max_length, 3, dtype= torch.float)
        scanpath[:valid_len] = torch.from_numpy(sp[:valid_len]) if isinstance(sp, np.ndarray) else sp[:valid_len]

        # image_feats
        img_feats_padded = torch.zeros(self.max_length, 2048, dtype=torch.float)
        img_feats_padded[:valid_len] = torch.from_numpy(sp_image_feats[:valid_len])

        # sem_feats
        sem_feats_padded = torch.zeros(self.max_length, 512, dtype=torch.float)
        sem_feats_padded[:valid_len] = torch.from_numpy(sp_sem_feats[:valid_len])

        # dec_input (shifted scanpath)
        dec_scan= torch.zeros(self.max_length, 3)
        dec_scan[1:valid_len] = scanpath[:valid_len-1]

        # dec_input for image_feats
        dec_image_feats = torch.zeros(self.max_length, 2048)
        dec_image_feats[1:valid_len] = img_feats_padded[:valid_len-1]

        # dec_input for sem_feats
        dec_sem_feats = torch.zeros(self.max_length, 512)
        dec_sem_feats[1:valid_len] = sem_feats_padded[:valid_len-1]

        # dec_mask
        dec_mask = torch.zeros(self.max_length)
        dec_mask[valid_len:] = 1

        sphere_coordinates = torch.randn(3, 10000)
        sphere_coordinates /= sphere_coordinates.norm(2, dim=0)
        
        return {
            'imgs': image,                  # The pre-extracted image tensor
            'scanpath': scanpath,           # [L, 3]
            'valid_len': valid_len,
            'dec_scan': dec_scan,
            'dec_image_feats': dec_image_feats, # shifted image feats
            'dec_sem_feats': dec_sem_feats,     # shifted sem feats
            'dec_masks': dec_mask,
            'csv_ids': sp_csv_id,           # ID of the testee
            'image_feats': img_feats_padded,# [L, 2048]
            'sem_feats': sem_feats_padded,  # [L, 512]
            'file_names': img_name,
            'sphere_coordinates': sphere_coordinates
        }

class EyeGaze360DataLoader(DataLoader):
    def __init__(self, pkl_path='data/datasets/eyegaze360.pkl', phase='train', batch_size=8, max_length=18, shuffle=True,seed=1218):
        self.seed = seed
        dataset = EyeGaze360Dataset(pkl_path, phase=phase, max_length=max_length)
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,drop_last=phase == "train", pin_memory=True,
                         worker_init_fn=self._init_fn)
    
    def _init_fn(self, worker_id):
        np.random.seed(int(self.seed) + worker_id)
