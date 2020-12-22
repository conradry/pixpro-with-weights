# PixPro -- Pixel-level representation learning

Unofficial implementation of [Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning](https://arxiv.org/abs/2011.10043).

<figure>
  <img src="./example.png"></img>
  <figcaption>Red and blue points show sampled locations in two image views. Large red dot marks a point in the first view and the large blue dots are "matches" for that pixel in the second view (based off a distance threshold of 0.7).</figcaption>
</figure>


## Current Status

Implementations of the dataloader, model and train_backbone script are complete for Pixel Propagation. Pixel Contrast and Pixel Propagation + Instance loss will be added in the future.

- [x] Pixel propagation module
- [x] Suppport for all spatial transforms (crops, resizing, flips, rotations, grid/elastic deformations, etc.)
- [x] Generic encoder and projection head for any torchvision model
- [x] Consistency loss for pixel propagation (not pixel contrast)
- [x] BYOL-style data augmentations
- [x] Cosine learning rate schedule
- [x] Momentum encoder's momentum schedule from BYOL (0.99 -> 1 during training)
- [x] LARS optimizer
- [x] Distributed training script for backbone network (e.g. resnet50)
- [ ] Additional instance level loss
- [ ] Pre-trained ResNet50 backbone model
- [ ] Pre-trained FPN
- [ ] Results on COCO and/or PASCAL

Hyperparameters and training schedules have been reproduced with as much fidelity to the original publication as possible.

On an 8 GPU machine, run:

```
python train_backbone.py {data_directory} {save_directory} -a resnet50 -b 1024 --lr 4 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
--world-size 1 --rank 0 --momentum 0.0
```

Where {data_directory} should be a path to a folder containing ImageNet training data.

For smaller batch sizes scale the learning rate by ```(lr = base_lr x batch_size/256)``` where ```base_lr=1```.