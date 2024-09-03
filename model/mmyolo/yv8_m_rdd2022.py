_base_ = "./configs/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py"

#
# https://mmengine.readthedocs.io/en/latest/api/visualization.html
#
_base_.visualizer.vis_backends = [
dict(type='LocalVisBackend'),
dict(type='TensorboardVisBackend'),
dict(type='WandbVisBackend', init_kwargs={
        'project': "rdd-mm",
        "reinit": True,}),]

#
# Train & Val - https://mmyolo.readthedocs.io/en/latest/get_started/15_minutes_object_detection.html
#
# ========================training configurations======================
work_dir = './work_dirs/yolov8_m_rdd'
max_epochs = 500
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 20
interval = 2
# Batch size of a single GPU during training
train_batch_size_per_gpu = 4
val_batch_size_per_gpu = train_batch_size_per_gpu

# -----data related-----
data_root = '/home/rahul/workspace/data/rdd/coco/'
# Path of train annotation file
train_ann_file = 'annotations/train.json'
train_data_prefix = 'train/images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/val.json'
val_data_prefix = 'val/images/'  # Prefix of val image path


class_names = ("D00", "D10", "D20", "D40", ) # dataset category name
num_classes = len(class_names)               # Number of classes for classification
# metainfo is a configuration that must be passed to the dataloader, otherwise it is invalid
# palette is a display color for category at visualization
# The palette length must be greater than or equal to the length of the classes
metainfo = dict(classes=class_names, palette=[[255,255,100], [255,200,200], [255,50,0], [200,200,0]])


# load COCO pre-trained weight
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth'  # noqa

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
)



# ========================modified parameters======================
deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

affine_scale = 0.9
mixup_prob = 0.15

# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.01
weight_decay = 0.0005

# =======================Unmodified in most cases==================
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform
mosaic_affine_transform = _base_.mosaic_affine_transform
last_transform = _base_.last_transform

# We also need to change the num_classes in head to match the dataset's annotation

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor
        ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]
        ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels])
        ),    
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes)
        )
    )

train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)), *last_transform
]

# RDD COCO data loader
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32), # Config of filtering images and annotations
        pipeline=train_pipeline
        ))
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32) # Config of filtering images and annotations
        ))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + val_ann_file, classwise=True, metric='bbox')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]


#
# TTA - https://mmyolo.readthedocs.io/en/latest/common_usage/tta.html
#
tta_img_scales = [(640, 640), (320, 320), (960, 960)]

tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))

_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=s),
            dict(
                type='LetterResize',
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]) for s in tta_img_scales
]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(type='mmdet.RandomFlip', prob=1.),
                dict(type='mmdet.RandomFlip', prob=0.)
            ], [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param', 'flip',
                               'flip_direction'))
            ]
        ])
]