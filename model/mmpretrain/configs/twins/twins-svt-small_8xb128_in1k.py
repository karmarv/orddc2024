_base_ = ['twins-svt-base_8xb128_in1k.py']

# model settings
model = dict(backbone=dict(arch='small'), head=dict(in_channels=512))
