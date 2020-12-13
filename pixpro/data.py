import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ContrastData(Dataset):
    def __init__(
        self,
        imdir,
        space_tfs,
        view1_color_tfs,
        view2_color_tfs=None
    ):
        super(ContrastData, self).__init__()

        self.imdir = imdir
        self.fnames = os.listdir(imdir)
        print(f'Found {len(self.fnames)} images in directory')

        #crops, resizes, flips, rotations, etc.
        self.space_tfs = space_tfs

        #brightness, contrast, jitter, blur, and
        #normalization
        self.view1_color_tfs = view1_color_tfs
        self.view2_color_tfs = view2_color_tfs

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fpath = os.path.join(self.imdir, self.fnames[idx])
        image = cv2.imread(fpath)

        y = np.arange(0, image.shape[0], dtype=np.float32)
        x = np.arange(0, image.shape[1], dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y, x)
        grid_y, grid_x = grid_y.T, grid_x.T

        #space transforms treat coordinate grid like an image
        #bilinear interp is good, nearest would be bad
        view1_data = self.space_tfs(image=image, grid_y=grid_y[..., None], grid_x=grid_x[..., None])
        view2_data = self.space_tfs(image=image, grid_y=grid_y[..., None], grid_x=grid_x[..., None])

        view1 = view1_data['image']
        view1_grid = np.concatenate([view1_data['grid_y'], view1_data['grid_x']], axis=-1)
        view2 = view2_data['image']
        view2_grid = np.concatenate([view2_data['grid_y'], view2_data['grid_x']], axis=-1)

        view1 = self.view1_color_tfs(image=view1)['image']
        if self.view2_color_tfs is not None:
            view2 = self.view2_color_tfs(image=view2)['image']
        else:
            view2 = self.view1_color_tfs(image=view2)['image']

        print(view1_grid.shape)

        output = {
            'fpath': fpath,
            'view1': view1,
            'view1_grid': torch.from_numpy(view1_grid).permute(2, 0, 1),
            'view2': view2,
            'view2_grid': torch.from_numpy(view2_grid).permute(2, 0, 1)
        }

        return output
