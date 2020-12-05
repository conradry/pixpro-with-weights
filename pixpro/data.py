import os
import cv2
from torch.utils.data import Dataset

class ContrastData(Dataset):
    def __init__(self, imdir, view1_tfs, view2_tfs=None):
        super(ContrastData, self).__init__()

        #list the images in the imdir
        self.imdir = imdir
        self.fnames = os.listdir(imdir)
        print(f'Found {len(self.fnames)} images in directory')

        #transforms in color/image space only
        #no crops!!!
        self.view1_tfs = view1_tfs
        self.view2_tfs = view2_tfs

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fpath = os.path.join(self.imdir, self.fnames[idx])
        image = cv2.imread(fpath)

        #two independent transforms
        view1 = self.view1_tfs(image=image)['image']
        if self.view2_tfs is not None:
            view2 = self.view2_tfs(image=image)['image']
        else:
            view2 = self.view1_tfs(image=image)['image']

        return {'fpath': fpath, 'view1': view1, 'view2': view2}
