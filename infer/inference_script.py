from mmdet.apis import DetInferencer


# Setup a checkpoint file to load
checkpoint = './best_coco_D00_precision_epoch_300.pth'

# Initialize the DetInferencer
inferencer = DetInferencer(model='rtmdet_l_8xb32-300e_coco', weights=checkpoint, device='cuda:0')

# Perform inference (Setup for 16GB target GPU memory)
#inferencer('demo/demo.jpg', out_dir='./output')
# /home/rahul/workspace/vision/orddc2024/dataset/rdd2022/coco/test/images/China_MotorBike_001978.jpg
# /home/rahul/workspace/vision/orddc2024/dataset/rdd2022/coco/test/images/United_States_006002.jpg
inferencer(['../dataset/rdd2022/coco/test/images/China_MotorBike_001978.jpg', '../dataset/rdd2022/coco/test/images/United_States_006002.jpg'], out_dir='./output')