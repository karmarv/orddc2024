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
