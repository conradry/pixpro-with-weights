# PixPro

Unofficial implementation of the code in: [Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning](https://arxiv.org/abs/2011.10043).

## Current Status

Implementations of the dataloader, model and train_backbone script are mostly complete for Pixel Propagation
(not Pixel Contrast).

Still left to do:

- [ ] Train ResNet50 backbone model
- [ ] Evaluate results on COCO and/or PASCAL
- [ ] Additional instance level loss
- [ ] Full FPN pre-training

Training now works with LARS (and presumably any other torch optimizer). Run:

```
python train_backbone.py {data_directory} {save_directory} -a resnet50 -b 1024 --lr 4 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
--world-size 1 --rank 0
```

Where {data_directory} should be a path to a folder containing ImageNet training data.

