import os
import cv2
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

def compute_distance_map(mask):
    """
    mask: (H, W), values in {0,1}
    return: (H, W) float distance map
    """
    mask = mask.astype(np.bool_)

    dist_out = distance_transform_edt(~mask)
    dist_in  = distance_transform_edt(mask)

    dist_map = dist_out + dist_in
    return dist_map

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, mask_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.mask_ids = mask_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def GetDist_map(self):
        os.makedirs(self.dist_dir, exist_ok=True)
        for name in tqdm(os.listdir(self.mask_dir)):
            mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)

            dist = compute_distance_map(mask)
            dist = np.clip(dist, 0, 20)  # ⭐ 非常推荐

            np.save(os.path.join(self.dist_dir, name.replace(".png", ".npy")), dist)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        mask_id = self.mask_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = cv2.imread(os.path.join(self.mask_dir, mask_id+self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']


        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask,{'img_id': img_id}
