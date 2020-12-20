## Pixpro: Transferring to Detection

Everything in this detection directory is copied from [Facebook's MoCo implementation](https://github.com/facebookresearch/moco/tree/master/detection). The only modification was to the ```convert-pretrain-to-detectron2.py``` script.

The `train_net.py` script reproduces the object detection experiments on Pascal VOC and COCO.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained PixPro model to detectron2's format:
   ```
   python convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

1. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_pixpro.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```

### Results

Below are the results on Pascal VOC 2007 test, fine-tuned on 2007+2012 trainval for 24k iterations using Faster R-CNN with a R50-C4 backbone:
