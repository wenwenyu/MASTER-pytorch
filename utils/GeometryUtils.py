import cv2
import numpy as np


def resize_with_height(_image, _target_height):
    """
    将图像高度resize到指定高度的等比例缩放

    Args:
        _image:     待缩放图像
        _target_height:     目标高度

    Returns:    缩放后的图像

    """
    h, w = _image.shape[:2]
    ratio = h / _target_height
    target_w = int(np.ceil(w / ratio))
    return cv2.resize(_image, (target_w, _target_height))


def _compute_image_specific_base(_image, _height_base=None, _width_base=None):
    """
    计算图像的宽高在一定基数基础上的最邻近向上取整的宽高

    Args:
        _image:     图像
        _height_base:   高度的基数
        _width_base:    宽度的基数

    Returns:    最临近高度，最邻近宽度

    """
    h, w = _image.shape[:2]
    target_h = h
    target_w = w
    if _height_base is not None:
        if h <= _height_base:
            target_h = _height_base
        else:
            target_h = int(np.ceil(h / _height_base) * _height_base)
    if _width_base is not None:
        if w <= _width_base:
            target_w = _width_base
        else:
            target_w = int(np.ceil(w / _width_base) * _width_base)
    return target_h, target_w


def pad_image_with_specific_base(_image,
                                 _left_margin, _top_margin,
                                 _height_base=None, _width_base=None,
                                 _pad_value=0,
                                 _output_pad_ratio=False):
    """
    将图像进行pad到特定尺寸上

    Args:
        _image:     待pad的图像
        _left_margin:   左边界比例
        _top_margin:    上边界比例
        _height_base:   高度的base
        _width_base:    宽度的base
        _pad_value:     pad的值
        _output_pad_ratio:  是否输出pad的占比

    Returns:    pad后的图像，（pad边界的占比）

    """
    h, w = _image.shape[:2]
    target_h, target_w = _compute_image_specific_base(_image, _height_base, _width_base)
    if len(_image.shape) == 3:
        full_size_image = np.ones((target_h, target_w, _image.shape[2]), dtype=_image.dtype) * _pad_value
    else:
        full_size_image = np.ones((target_h, target_w), dtype=_image.dtype) * _pad_value
    right_margin = _left_margin + w
    bottom_margin = _top_margin + h
    full_size_image[_top_margin:bottom_margin, _left_margin:right_margin, ...] = _image
    if not _output_pad_ratio:
        return full_size_image
    else:
        return full_size_image, (_left_margin / target_w, _top_margin / target_h)
