_base_ = [
    'mmpretrain::_base_/datasets/imagenet_bs256_rsb_a12.py',
    'mmpretrain::_base_/schedules/imagenet_bs2048_rsb.py',
    'mmpretrain::_base_/default_runtime.py'
]

# https://mmengine.readthedocs.io/en/latest/api/visualization.html
_base_.visualizer.vis_backends = [
dict(type='LocalVisBackend'),
dict(type='TensorboardVisBackend'),
dict(type='WandbVisBackend', init_kwargs={
        'project': "rdd-mm",
        "reinit": True,}),]


max_epochs = 600
data_root='/home/rahul/workspace/vision/orddc2024/dataset/rdd2022/imgnet/'


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        out_indices=(4, ),
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='mmdet.SiLU')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.2),
        dict(type='CutMix', alpha=1.0)
    ]))

# dataset settings

# >>>>>>>>>>>>>>> Override dataset settings here >>>>>>>>>>>>>>>>>>>
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=64,
    sampler=dict(type='RepeatAugSampler', shuffle=True),
    dataset=dict(
        type='CustomDataset',
        data_root=data_root+"train",
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='',    # The `data_root` is the data_prefix directly.
        with_label=True,
        _delete_=True,
        pipeline=train_pipeline  # Need to specify pipeline
    )
)



val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),  # Scale the short side to 256
    dict(type='CenterCrop', crop_size=224),     # center crop
    dict(type='PackInputs')
]

val_dataloader = dict(
    batch_size=64,
    sampler=dict(type='RepeatAugSampler', shuffle=True),
    dataset=dict(
        type='CustomDataset',
        data_root=data_root+"val",
        ann_file='',          # We assume you are using the sub-folder format without ann_file
        data_prefix='',       # The `data_root` is the data_prefix directly.
        with_label=True,
        #split='train',
        _delete_=True,
        pipeline=val_pipeline  # Need to specify pipeline
    )
)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(weight_decay=0.01),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs-10,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=max_epochs)
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)
default_hooks = dict(
    # only keeps the latest checkpoints
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=100))
