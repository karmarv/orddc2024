# projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py
_base_ = './projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py'


# https://mmengine.readthedocs.io/en/latest/api/visualization.html
_base_.visualizer.vis_backends = [
dict(type='LocalVisBackend'),
dict(type='TensorboardVisBackend'),
dict(type='WandbVisBackend', init_kwargs={
        'project': "rdd-mm",
        "reinit": True,}),]


# -----data related-----
data_root = '/home/rahul/workspace/vision/orddc2024/dataset/rdd2022/coco/'
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

# model settings

# Batch size of a single GPU during training
train_batch_size_per_gpu = 2
val_batch_size_per_gpu = train_batch_size_per_gpu

max_epochs = 15

# load COCO pre-trained weight
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_lsj_8xb2_1x_coco/co_dino_5scale_r50_lsj_8xb2_1x_coco-69a72d67.pth'  # noqa


# Model setup
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    use_lsj=False, 
    data_preprocessor=dict(pad_mask=False, batch_augments=None)
    )

# Pipelines

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# COCO data loader
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        _delete_=True,
        type=_base_.dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=_base_.backend_args
        )
    )
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        _delete_=True,
        type=_base_.dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=test_pipeline,
        backend_args=_base_.backend_args
        )
    )
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(proposal_nums=(100, 1, 10), ann_file=data_root + val_ann_file, classwise=True)
test_evaluator = val_evaluator


train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[14],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=50))
log_processor = dict(by_epoch=True)
