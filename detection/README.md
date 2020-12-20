## Pixpro: Transferring to Detection

Everything in this detection directory is copied from [Facebook's MoCo implementation](https://github.com/facebookresearch/moco/tree/master/detection). The only modification was to the ```convert-pretrain-to-detectron2.py``` script.

### Instructions

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
