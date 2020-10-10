Training
#####
#SUN-RGBD
CUDA_VISIBLE_DEVICES=0 python train.py --dataset sunrgbd --log_dir log_sunrgbd


python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

# ScanNet
CUDA_VISIBLE_DEVICES=0 python train.py --dataset scannet --log_dir log_scannet --num_point 40000

python eval.py --dataset scannet --checkpoint_path log_scannet/checkpoint.tar --dump_dir eval_scannet --num_point 40000 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

