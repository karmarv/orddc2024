_base_ = './cspnext-s_8xb256-rsb-a1-600e_in1k.py'

model = dict(
    backbone=dict(deepen_factor=1, widen_factor=1),
    head=dict(in_channels=1024))


max_epochs = 600
#data_root='/home/rahul/workspace/vision/orddc2024/dataset/rdd2022/imgnet/'
data_root='/home/rahul/workspace/vision/rdd/orddc2024/dataset/rdd2022/imgnet/'
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
    batch_size=384,
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
    batch_size=96,
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
        T_max=max_epochs-5,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=max_epochs)
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)
default_hooks = dict(
    # only keeps the latest checkpoints
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=100))
