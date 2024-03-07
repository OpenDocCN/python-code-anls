# `.\YOLO-World\configs\deploy\detection_onnxruntime-int8_dynamic.py`

```
# 设置基础路径为指定的配置文件路径
_base_ = (
    '../../third_party/mmdeploy/configs/mmdet/detection/'
    'detection_onnxruntime-fp16_dynamic.py')
# 定义后端配置，设置精度为int8
backend_config = dict(
    precision='int8')
# 定义代码库配置，包括模型类型、任务类型、后处理参数等
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.1,
        confidence_threshold=0.005,
        iou_threshold=0.3,
        max_output_boxes_per_class=100,
        pre_top_k=1000,
        keep_top_k=100,
        background_label_id=-1),
    module=['mmyolo.deploy'])
# 重新定义后端配置，设置类型为onnxruntime
backend_config = dict(
    type='onnxruntime')
```