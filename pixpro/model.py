import random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: For syncbatchnorm
#process_group = torch.distributed.new_group(process_ids)
#sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
#TODO: Add device for pair_distance
#TODO: Is PPM applied before or after resampling? Probably after, right?
#TODO: Test on GCP
#TODO: Visualize some positive pairs to see if they make sense

def resample(image, crop_box):
    """
    Resizes (warps) a cropped view to its original size in the image
    """
    y1, x1, y2, x2 = crop_box
    crop_height = y2 - y1
    crop_width = x2 - x1

    #TODO: should it be bilinear or nearest here?
    #together with PPM does bilinear make representations too smooth?
    return F.interpolate(image, size=(crop_height, crop_width), mode='bilinear', align_corners=True)

def ravel_index(row, col, ncols):
    """
    Like the numpy function (sort of)
    """
    return (row * ncols - 1) + (col + 1)

def positive_pairs(view1_crop_box, view2_crop_box, distance_thr=32, device='cpu'):
    """
    Finds positives pairs of pixels within the 2 given views. A
    positive pair is any pair within distance_thr of each other.

    Returns:
    --------
    view1_raveled_indices: List of indices for the raveled version
    of view1. Each item is a single index.

    view2_positive_pairs: List of indices in view2 that are positive
    matches for each pixel in view1_raveled_indices. Each item has
    at least 1 element.
    """

    #TODO create the distance matrix on device?
    oy1, ox1, oy2, ox2 = view1_crop_box
    ty1, tx1, ty2, tx2 = view2_crop_box

    #calculate pairwise y and x distances
    view1_y = torch.arange(oy1, oy2, dtype=torch.float32, device=device)[:, None] #(OH, 1)
    view2_y = torch.arange(ty1, ty2, dtype=torch.float32, device=device)[None, :] #(1, TH)
    distance_y = view1_y - view2_y #(OH, TH)

    view1_x = torch.arange(ox1, ox2, dtype=torch.float32, device=device)[:, None] #(OW, 1)
    view2_x = torch.arange(tx1, tx2, dtype=torch.float32, device=device)[None, :] #(1, TW)
    distance_x = view1_x - view2_x #(OW, TW)

    #calculate the actual distances between possible pairs
    v1h, v1w = view1_y.size(0), view1_x.size(0)
    v2w = view2_x.size(1)
    v1_raveled_index = []
    v2_positive_matches = []
    for i in range(v1w):
        for j in range(v1h):
            #(TH, 1) + (1, WH) --> (TH, WH)
            v2_distances = torch.sqrt(
                distance_y[j, :, None] ** 2 + distance_x[i, None, :] ** 2
            )
            v2_raveled_indices = torch.where(v2_distances < distance_thr)

            if len(v2_raveled_indices[0]) > 0:
                index = ravel_index(j, i, v1w)
                v1_raveled_index.append(index)
                v2_raveled_indices = ravel_index(*v2_raveled_indices, v2w)
                v2_positive_matches.append(v2_raveled_indices)

    return v1_raveled_index, v2_positive_matches

#create a function to extract two crops from
#an image batch (B, C, H, W)
class CutoutViews(nn.Module):
    def __init__(
        self,
        height,
        width,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interp='bilinear',
        align_corners=True
    ):
        super(CutoutViews, self).__init__()

        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio

        #create the interpolator
        self.upsample = nn.Upsample(size=(height, width), mode=interp, align_corners=align_corners)

    def get_crop_parameters(self, image_height, image_width):
        #a copy from albumentations with some small modifications
        area = image_height * image_width

        for _attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= image_width and 0 < h <= image_height:
                i = random.randint(0, image_height - h)
                j = random.randint(0, image_width - w)
                h_start = i * 1.0 / (image_height - h + 1e-10)
                w_start = j * 1.0 / (image_width - w + 1e-10)

                y1 = max(0, int((image_height - h) * h_start))
                y2 = min(y1 + h, image_height - 1)
                x1 = max(0, int((image_width - w) * w_start))
                x2 = min(x1 + w, image_width - 1)
                return y1, x1, y2, x2

        # Fallback to central crop
        in_ratio = image_width / image_height
        if in_ratio < min(self.ratio):
            w = image_width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = image_height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = image_width
            h = image_height
        i = (image_height - h) // 2
        j = (image_width - w) // 2

        h_start = i * 1.0 / (image_height - h + 1e-10)
        w_start = j * 1.0 / (image_width - w + 1e-10)
        y1 = max(0, int((image_height - self.height) * h_start))
        y2 = min(y1 + self.height, image_height - 1)
        x1 = max(0, int((image_width - self.width) * w_start))
        x2 = min(x1 + self.width, image_width - 1)
        return y1, x1, y2, x2

    def forward(self, view1, view2):
        #while the crops are randomly chosen, they are applied
        #uniformly across all images, this keeps all the tensors
        #in nice evenly sized blocks that we can manage more easily

        assert(view1.shape == view2.shape), "view1 and view2 have different shapes!"
        image_height, image_width = view1.size()[2:]
        oy1, ox1, oy2, ox2 = self.get_crop_parameters(image_height, image_width)
        ty1, tx1, ty2, tx2 = self.get_crop_parameters(image_height, image_width)

        #calculate the overlap between the boxes
        ymin = max(oy1, ty1)
        xmin = max(ox1, tx1)
        ymax = min(oy2, ty2)
        xmax = min(ox2, tx2)
        overlap = (ymax - ymin) * (xmax - xmin)
        #print(f'batch overlap: {overlap / (self.height * self.width)}')

        #apply the crops
        view1 = view1[..., oy1:oy2, ox1:ox2]
        view2 = view2[..., ty1:ty2, tx1:tx2]

        #resize back to original image shape to get
        #two tensors of (B, C, H, W)
        view1 = self.upsample(view1)
        view2 = self.upsample(view2)

        return view1, view2, (oy1, ox1, oy2, ox2), (ty1, tx1, ty2, tx2)

class PPM(nn.Module):
    def __init__(self, nin, gamma=2, nlayers=1):
        super(PPM, self).__init__()

        #the networks can have 0-2 layers
        if nlayers == 0:
            layers = [nn.Identity()]
        elif nlayers == 1:
            layers = [nn.Conv2d(nin, nin, 1)]
        elif nlayers == 2:
            layers = [
                nn.Conv2d(nin, nin, 1, bias=False),
                nn.BatchNorm2d(nin),
                nn.ReLU(),
                nn.Conv2d(nin, nin, 1)
            ]
        else:
            raise Exception(f'nlayers must be 0, 1, or 2, got {nlayers}')

        self.transform = nn.Sequential(*layers)
        self.gamma = gamma
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        #(B, C, H, W, 1, 1) x (B, C, 1, 1, H, W) --> (B, H, W, H, W)
        #relu is same as max(sim, 0) but differentiable
        xi = x[:, :, :, :, None, None]
        xj = x[:, :, None, None, :, :]
        s = F.relu(self.cosine_sim(xi, xj)) ** self.gamma

        #output of "g" in the paper (B, C, H, W)
        gx = self.transform(x)

        #use einsum and skip all the transposes
        #and matrix multiplies
        y = torch.einsum('bijhw, bchw -> bcij', s, gx)

        return x

class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()

        #strip off models fc layer (always last for torchvision models)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        #wonky way to get the numbers of channels in the output
        #but it does work for pretty much any model
        projection_nin = list(self.backbone.named_parameters())[-1][1].shape[0]

        #note that projection_nin = 2048 for resnet50 (matches paper)
        self.projection = nn.Sequential(
            nn.Conv2d(projection_nin, projection_nin, 1, bias=False),
            nn.BatchNorm2d(projection_nin),
            nn.ReLU(inplace=True),
            nn.Conv2d(projection_nin, 256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.projection(x)

class PixPro(nn.Module):
    def __init__(
        self,
        backbone,
        crop_size=(224, 224),
        gamma=2,
        ppm_layers=1,
        momentum=0.99,
        distance_thr=32
    ):
        super(PixPro, self).__init__()

        #create the encoder and momentum encoder
        self.encoder = Encoder(backbone)
        self.mom_encoder = Encoder(backbone)

        #hardcoded: encoder outputs 256
        self.ppm = PPM(256)

        #copy parameters from the encoder to momentum encoder
        #and turn off gradients
        for param, mom_param in zip(self.encoder.parameters(), self.mom_encoder.parameters()):
            mom_param.data.copy_(param.data)
            mom_param.requires_grad = False

        self.cutout = CutoutViews(*crop_size)
        self.momentum = momentum
        self.distance_thr = distance_thr
        #momentum = 1 − (1 − momentum) * (cos(pi * k / K) + 1) / 2 #k current step

    @torch.no_grad()
    def update_mom_encoder(self):
        for param, mom_param in zip(self.encoder.parameters(), self.mom_encoder.parameters()):
            mom_param.data = mom_param.data * self.momentum + param.data * (1. - self.momentum)


    def forward(self, view1, view2):
        #crop the two views
        view1, view2, v1_box, v2_box = self.cutout(view1, view2)

        #pass each view through each encoder
        y = self.ppm(self.encoder(view1))
        yp = self.ppm(self.encoder(view2))

        with torch.no_grad():
            z = self.mom_encoder(view1)
            zp = self.mom_encoder(view2)

        #resample (warp) views back to their sizes
        #in the original image space
        y = resample(y, v1_box)
        yp = resample(yp, v2_box)
        z = resample(z, v1_box)
        zp = resample(zp, v2_box)

        #determine pairs of pixels to evaluate consistency loss
        v1_indices, v2_pairs = \
        positive_pairs(v1_box, v2_box, self.distance_thr, y.device)

        return y, yp, z, zp, view1, view2, v1_box, v2_box, v1_indices, v2_pairs

class ConsistencyLoss(nn.Module):
    def __init__(self, distance_thr=0.7*45):
        super(ConsistencyLoss, self).__init__()
        self.distance_thr = distance_thr
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, y, yp, z, zp, distances):
        #view1_shape = (H1, W1)
        #view2_shape = (H2, W2)
        H1, W1, H2, W2 = distances.size()
        distances = distances.reshape(H1 * W1, H2 * W2)
        #positive_mask = distances < self.distance_thr
        v1_pos, v2_pos = torch.where(distances < self.distance_thr)
        del distances

        #no positive pairs (should be rare)
        if len(v1_pos) == 0:
            return None

        #(B, C, H1 * W1) or (B, C, H2 * W2)
        y = y.flatten(2, -1)
        yp = yp.flatten(2, -1)
        z = z.flatten(2, -1)
        zp = zp.flatten(2, -1)

        #TODO: make this non-loopy without using too much memory?
        bsz = y.size(0)
        lpix = torch.zeros((bsz,))
        for v1_index in torch.unique(v1_pos):
            v2_indices = v2_pos[v1_pos == v1_index] #(k,)

            #(B, C, 1) x (B, C, k) --> (B, k)
            cos_y_zp = (self.cosine_sim(y[..., v1_index][..., None],  zp[..., v2_indices]))
            cos_z_yp = (self.cosine_sim(z[..., v1_index][..., None],  yp[..., v2_indices]))
            lpix += (-1 * (cos_y_zp + cos_z_yp)).sum(-1) #(B,)

        #average by total positives per image
        #then over the batch
        return (lpix / len(v2_pos)).mean() #lpix.mean()
