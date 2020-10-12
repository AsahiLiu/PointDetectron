# Poinnt Detectron
Created by Xu Liu, from <a href="https://air.jd.com/" target="_blank">JD AI Research</a> and <a href="https://www.u-tokyo.ac.jp/focus/ja/tags/?tag=UTOKYO%20VOICES" target="_blank">The University of Tokyo</a>.

![teaser](https://github.com/AsahiLiu/PointDetectron/tree/main/doc/NIPS_new.jpeg)

## Introduction
This repository is code release for our NeurIPS 2020 paper Group Contextual Encoding for 3D Poit Clouds (arXiv report [here](https://arxiv.org/pdf/)) and 3DV 2020 paper Dense Point Diffusion for 3D Detection (arXiv report [here](https://arxiv.org/pdf/))

This repository built on the VoteNet, we empower VoteNet model with Group Contextual Encoding Block, Dense Point Diffusion modules as well as the Dilated Point Convolution.
## Citation



## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/) and [Tensorflow](https://github.com/tensorflow/tensorflow) (for TensorBoard). It is required that you have access to GPUs. Matlab is required to prepare data for SUN RGB-D. The code is tested with Ubuntu 18.04, Pytorch v1.1, TensorFlow v1.14, CUDA 10.0 and cuDNN v7.4. Note: there is some incompatibility with newer version of Pytorch (e.g. v1.3), which is to be fixed.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

To see if the compilation is successful, try to run `python models/votenet.py` to see if a forward pass works.

Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    torch-encoding
    plyfile
    'trimesh>=2.35.39,<2.35.40'

## Run demo

Following VoteNet, you can download pre-trained VoteNet models and sample point clouds [HERE](https://drive.google.com/file/d/1oem0w5y5pjo2whBhAqTtuaYuyBu1OG8l/view?usp=sharing). Unzip the file under the project root path (`/path/to/project/demo_files`) and then run:

    python demo.py

The demo uses a pre-trained model (on SUN RGB-D) to detect objects in a point cloud from an indoor room of a table and a few chairs (from SUN RGB-D val set). You can use 3D visualization software such as the [MeshLab](http://www.meshlab.net/) to open the dumped file under `demo_files/sunrgbd_results` to see the 3D detection output. Specifically, open `***_pc.ply` and `***_pred_confident_nms_bbox.ply` to see the input point cloud and predicted 3D bounding boxes.

You can also run the following command to use another pretrained model on a ScanNet:

    python demo.py --dataset scannet --num_point 40000

Detection results will be dumped to `demo_files/scannet_results`.

## Training and evaluating

### Data preparation

For SUN RGB-D, follow the [README](https://github.com/facebookresearch/votenet/blob/master/sunrgbd/README.md) under the `sunrgbd` folder.

For ScanNet, follow the [README](https://github.com/facebookresearch/votenet/blob/master/scannet/README.md) under the `scannet` folder.

### Train and test on SUN RGB-D

To train a new  model ${MODEL_CONFIG} in the MODEL ZOO on SUN RGB-D data (depth images):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset sunrgbd --log_dir log_sunrgbd --model ${MODEL_CONFIG}

You can use `CUDA_VISIBLE_DEVICES=0,1,2` to specify which GPU(s) to use. Without specifying CUDA devices, the training will use all the available GPUs and train with data parallel (Note that due to I/O load, training speedup is not linear to the nubmer of GPUs used). 
While training you can check the `log_sunrgbd/log_train.txt` file on its progress, or use the TensorBoard to see loss curves.

To test the trained model with its checkpoint:

    python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal  --model  ${MODEL_CONFIG}

Example results will be dumped in the `eval_sunrgbd` folder (or any other folder you specify). You can run `python eval.py -h` to see the full options for evaluation. After the evaluation, you can use MeshLab to visualize the predicted votes and 3D bounding boxes (select wireframe mode to view the boxes).
Final evaluation results will be printed on screen and also written in the `log_eval.txt` file under the dump directory. In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on oriented boxes. 

### Train and test on ScanNet

To train a  model ${MODEL_CONFIG} in the MODEL ZOO on Scannet data (fused scan):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset scannet --log_dir log_scannet --num_point 40000 --model  ${MODEL_CONFIG}

To test the trained model with its checkpoint:

    python eval.py --dataset scannet --checkpoint_path log_scannet/checkpoint.tar --dump_dir eval_scannet --num_point 40000 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal --model  ${MODEL_CONFIG}
 
Example results will be dumped in the `eval_scannet` folder (or any other folder you specify). 

### MODEL ZOO

|        MODEL SPECS                 |     $ {MODEL_CONFIG}          | SUN-RGBD | ScanNet |
|---------------------------------------------|----------:|----------:|:-------:|
| [Group Contextual Ecoding (K=8, G=12, C×3)](models/votenet_enc_FP2_K8_G12_C3.py)|votenet_enc_FP2_K8_G12_C3  | 60.7 | 60.8 |
| [SA2 - Dense Point Diffusion (3,6,12)](models/votenet_SA2_denseaspp3_6_12.py) |votenet_SA2_denseaspp3_6_12| 58.6 | 59.6 |
| [SA2 - Dense Point Diffusion (3,6)](models/votenet_SA2_denseaspp3_6.py)|votenet_SA2_denseaspp3_6| 58.7 | 58.9 |
| [VoteNet](models/votenet.py) | votenet (default)| 57.7 | 58.6 |



The ablation models in the papers can be derived from the models listed above, therefore, we did not list them all to save the space.
### Train on your own data

[For Pro Users] If you have your own dataset with point clouds and annotated 3D bounding boxes, you can create a new dataset class and train VoteNet on your own data. To ease the proces, some tips are provided in this [doc](https://github.com/facebookresearch/votenet/blob/master/doc/tips.md).

## Acknowledgements
We want to thank Charles Qi for his VoteNet ([original codebase](https://github.com/facebookresearch/votenet)), Hang Zhang for his EncNet ([original codebase](https://hangzhang.org/PyTorch-Encoding/)) and  Erik Wijmans for his PointNet++ implementation in Pytorch ([original codebase](https://github.com/erikwijmans/Pointnet2_PyTorch)).

## License
votenet is relased under the MIT License. See the [LICENSE file](https://arxiv.org/pdf/1904.09664.pdf) for more details.
