# PixPro

Unofficial implementation of the code in: [Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning](https://arxiv.org/abs/2011.10043).

## Current Status

Implementations of the dataloader, model and train_backbone script are mostly complete for Pixel Propagation
(not Pixel Contrast).

Still left to do:

- [ ] Find/write a working implementation of LARS
- [ ] Train ResNet50 backbone model
- [ ] Evaluate results on COCO and/or PASCAL
- [ ] Additional instance level loss
- [ ] Full FPN pre-training

Training currently works for SGD (and presumably any other torch optimizer). Run:

```
python train_backbone.py {data_directory} {save_directory} -a resnet50 -b 1024 --lr 0.06 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
--world-size 1 --rank 0
```

Where {data_directory} should be a path to a folder containing ImageNet training data. Note that the lr would need to be tested for SGD. 0.06 is a guess based off the LR used by MoCo.

