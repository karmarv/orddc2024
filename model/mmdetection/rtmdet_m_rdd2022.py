# model/mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py
_base_ = './configs/rtmdet/rtmdet_m_8xb32-300e_coco.py'


# https://mmengine.readthedocs.io/en/latest/api/visualization.html
_base_.visualizer.vis_backends = [
dict(type='LocalVisBackend'),
dict(type='TensorboardVisBackend'),
dict(type='WandbVisBackend', init_kwargs={
        'project': "rdd-mm",
        "reinit": True,}),]

# ========================training configurations======================
work_dir = './work_dirs/rtmdet_m_rdd_scale_plus'
max_epochs = 300
stage2_num_epochs = 50
base_lr = 0.001
interval = 5

# Batch size of a single GPU during training
train_batch_size_per_gpu = 4
val_batch_size_per_gpu = train_batch_size_per_gpu

# -----data related-----
data_root = '/home/rahul/workspace/vision/orddc2024/dataset/rdd2022/coco_plus/'
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

# Model setup

# load COCO pre-trained weight
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'  # noqa
# mmpretrain cspnext-l checkpoint
#checkpoint =  "../mmpretrain/work_dirs/cspnext-m_8xb256-rsb-a1-600e_in1k/20240903_094527/epoch_600.pth"

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    #backbone=dict(
    #    # Since the checkpoint includes CUDA:0 data,
    #    # it must be forced to set map_location.
    #    # Once checkpoint is fixed, it can be removed.
    #    init_cfg=dict(
    #        type='Pretrained',
    #        prefix='backbone.',
    #        checkpoint=checkpoint)
    #    ),     
    bbox_head=dict(
        num_classes=num_classes
        )
    )

# ========================modified parameters======================

# Pipelines
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',        # multiscale by variable resize - https://mmcv.readthedocs.io/en/2.x/_modules/mmcv/transforms/processing.html#RandomResize
        scale=[(640, 640), (1280, 1280)],
        ratio_range=(0.5, 2.0),
        keep_ratio=True),        
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# COCO data loader
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        pipeline=train_pipeline
        )
    )
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)
        )
    )
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(proposal_nums=(100, 1, 10), ann_file=data_root + val_ann_file, classwise=True)
test_evaluator = val_evaluator


# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=30  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]


# TTA
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

img_scales = [(640, 640), (960, 960), (1280, 1280)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=s, keep_ratio=True)
                for s in img_scales
            ],
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [
                dict(
                    type='Pad',
                    size=(960, 960),
                    pad_val=dict(img=(114, 114, 114))),
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]