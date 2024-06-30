import asyncio
from argparse import ArgumentParser
import cv2
import sys
sys.path.append("/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR")
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from projects import *

config_file = '/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR/projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/instance_imagenav/Object-Goal-Navigation/3rdparty/CoDETR/checkpoints/co_dino_5scale_swin_large_16e_o365tococo.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img = cv2.imread('/instance_imagenav/Object-Goal-Navigation/envs/utils/demo.jpg', 0)
img = '/instance_imagenav/Object-Goal-Navigation/envs/utils/demo.jpg'
result = inference_detector(model, img)
# print(result)



'''
56 : chair
57: couch
58: potted_plant
59: bed
61: toilet
62: tv
result format:
(80, len(predicted), 5)
(5 means: xyxy, logits)
computation costs: 7427 mb


'''