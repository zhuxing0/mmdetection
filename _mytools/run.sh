python tools/train.py configs/_myconfigs/faster_rcnn_r50_fpn_2x_step_taskdata.py --gpu-id 2

python tools/train.py configs/_myconfigs/faster_rcnn_r50_fpn_2x_coslr_taskdata.py --gpu-id 2


python tools/test.py configs/_myconfigs/faster_rcnn_r50_fpn_2x_step_taskdata.py work_dirs/faster_rcnn_r50_fpn_2x_step_taskdata/epoch_24.pth --format-only --options "jsonfile_prefix=./work_dirs/faster_rcnn_r50_fpn_2x_step_taskdata/result" --gpu-id 2

python tools/test.py configs/_myconfigs/faster_rcnn_r50_fpn_2x_coslr_taskdata.py work_dirs/faster_rcnn_r50_fpn_2x_coslr_taskdata/epoch_24.pth --format-only --options "jsonfile_prefix=./work_dirs/faster_rcnn_r50_fpn_2x_coslr_taskdata/result"

python tools/dataset_converters/images2coco.py _mydatasets/data_image_processing_task/test/ _mydatasets/data_image_processing_task/class.txt annotations/test_coco.json

python tools/train.py configs/_myconfigs/DAC_yolox_s.py --gpu-id 1
python tools/train.py configs/_myconfigs/DAC_yolov3.py --gpu-id 1

python tools/train.py configs/_myconfigs/DAC_fcos_r50.py --gpu-id 2

python tools/train.py configs/_myconfigs/DAC_atss_r50.py --gpu-id 3

python tools/test.py configs/_myconfigs/DAC_atss_r50.py work_dirs/DAC_atss_r50/latest.pth --eval bbox 

python tools/test.py configs/_myconfigs/DAC_fcos_r50.py work_dirs/DAC_fcos_r50/latest.pth --eval bbox 