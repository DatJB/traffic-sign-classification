from pathlib import Path
import cv2
import numpy as np


def read_bgr(path):
    """Unicode-safe image decoding, including original GTSRB PPM files."""
    path = Path(path)
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f'Empty image file: {path}')
    try:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f'Cannot decode image: {path}') from exc
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def resize_bgr(image, size=(64, 64)):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 BGR image with shape (H, W, 3)')
    if len(size) != 2 or any(int(v) != v or v <= 0 for v in size):
        raise ValueError('size must be positive integer (width, height)')
    interpolation = (cv2.INTER_AREA if image.shape[1] >= size[0]
                     and image.shape[0] >= size[1] else cv2.INTER_LINEAR)
    return cv2.resize(image, tuple(map(int, size)), interpolation=interpolation)


def preprocess(image, size=(64, 64), color='bgr'):
    """Resize full supplied crop (no ROI crop); no learned normalization.

    OpenCV uint8 HSV ranges: H=0..179, S/V=0..255.
    Gray is intended for HOG; HSV retains color for histograms.
    """
    image = resize_bgr(image, size)
    conversions = {'hsv': cv2.COLOR_BGR2HSV, 'gray': cv2.COLOR_BGR2GRAY,
                   'rgb': cv2.COLOR_BGR2RGB}
    if color == 'bgr':
        return image
    if color not in conversions:
        raise ValueError(f'Unsupported color: {color}')
    return cv2.cvtColor(image, conversions[color])
