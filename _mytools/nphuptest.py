# -*- coding:utf-8 -*-
import time


def sleep_time(time_hour, time_min, time_second):
    return time_hour * 3600 + time_min * 60 + time_second

from mmdet.apis import init_detector, inference_detector
import mmcv

import json
import os

if __name__ == '__main__':
    num = 0
    second = sleep_time(0, 0, 2)

    # 指定模型的配置文件和 checkpoint 文件路径
    config_file = 'configs/_myconfigs/faster_rcnn_r50_fpn_2x_coslr_taskdata.py'
    checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_2x_coslr_taskdata/epoch_24.pth'

    # 根据配置文件和 checkpoint 文件构建模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_dir = '_mydatasets/data_image_processing_task/test'
    name_list = os.listdir(img_dir)

    upload_json = []

    for i in range(len(name_list)):
        name = os.path.join(img_dir, '%04d' % (i+1) + ".jpg")

        # 测试单张图片并展示结果
        img = mmcv.imread(name)  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次

        result = inference_detector(model, img)
        # print(result[0])

        result = result[0].tolist()

        for j in range(len(result)):
            dt = {}
            dt['name'] = '%04d' % (i+1) + ".jpg"

            print(dt['name'])

            dt['category_id'] = 1

            bbox = []
            bbox.append(round(result[j][0], 2))
            bbox.append(round(result[j][1], 2))
            bbox.append(round(result[j][2] - result[j][0], 2))
            bbox.append(round(result[j][3] - result[j][1], 2))

            dt['bbox'] = bbox
            dt['score'] = result[j][4]
            # print(dt)
            upload_json.append(dt)

            time.sleep(second)



