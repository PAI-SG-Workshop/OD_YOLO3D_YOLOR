"""
YOLOR builders for the three tasks (anchor-based).

Paper : Wang, Yeh & Liao, *You Only Learn One Representation: Unified Network for Multiple Tasks*, arXiv:2105.04206
Code  : official Darknet cfg (see the cfg headers), built by the common Darknet builder.
Delta : YOLOR-CSP - Scaled-YOLOv4 CSP body + implicit knowledge (implicit_add / implicit_mul vectors added to / multiplied with the head features), SiLU
Detection layers: reused unchanged from detectors/yolov3 (as v2 / v4 / v5 / v7)
    image -> Image YOLO layer, complex -> Complex-YOLO layer, yolo3d -> YOLO3D layer
Not used (same policy as v4 / v7): scale_x_y grid sensitivity, CIoU, Mosaic and the other bag-of-freebies.
"""

from pathlib import Path

from common.darknet import Darknet
from detectors.yolov3.build import yolo3d_factory

CFG_DIR = Path(__file__).resolve().parent / "cfg"
SPECS = {
    "yolo3d": ("yolo3d_yolor.cfg", yolo3d_factory, 8),
}


def build_model(task, num_classes):
    cfg_name, factory, num_box_attr = SPECS[task]
    return Darknet(CFG_DIR / cfg_name, factory, num_classes, num_box_attr)
