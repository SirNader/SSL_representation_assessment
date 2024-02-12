from functools import partial

from kappadata import KDComposeTransform
from kappadata.transforms import ImageRangeNorm
from torchvision.transforms import Compose, Normalize


def flatten_transforms(transform):
    if transform is None:
        return []
    if isinstance(transform, KDComposeTransform):
        result = []
        for t in transform.transforms:
            result += flatten_transforms(t)
        return result
    return [transform]


def extract_denormalization_transform(x_transform, inplace=False):
    transforms = flatten_transforms(x_transform)
    norm_transforms = [transform for transform in transforms if isinstance(transform, (Normalize, ImageRangeNorm))]
    if len(norm_transforms) == 0:
        return None
    assert len(norm_transforms) == 1
    norm_transform = norm_transforms[0]
    if isinstance(norm_transform, ImageRangeNorm):
        return partial(norm_transform.denormalize, inplace=inplace)
    return Compose([
        Normalize(mean=(0., 0., 0.), std=tuple(1 / s for s in norm_transform.std)),
        Normalize(mean=tuple(-m for m in norm_transform.mean), std=(1., 1., 1.)),
    ])
