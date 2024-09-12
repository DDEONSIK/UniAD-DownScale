from mmcv.utils.registry import Registry, build_from_cfg

print("추적 UniAD/projects/mmdet3d_plugin/datasets/samplers/sampler.py 지나감")


SAMPLER = Registry('sampler')


def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, SAMPLER, default_args)
