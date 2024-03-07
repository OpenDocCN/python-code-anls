# `.\YOLO-World\configs\deploy\detection_tensorrt-fp16_static-640x640.py`

```
_base_ = (
    '../../third_party/mmyolo/configs/deploy/'
    'detection_tensorrt-fp16_static-640x640.py')
# 设置基础配置文件路径

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['dets', 'labels'],
    input_shape=(640, 640),
    optimize=True)
# 设置 ONNX 配置参数，包括类型、是否导出参数、是否保留初始化器作为输入、操作集版本、保存文件名、输入输出名称、输入形状、是否优化

backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=True, max_workspace_size=1 << 34),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 640, 640])))
    ])
# 设置后端配置参数，包括类型、通用配置、模型输入

use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501
# 是否使用 EfficientNMS 插件替换 TRTBatchedNMS 插件

codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.25,
        confidence_threshold=0.005,
        iou_threshold=0.65,
        max_output_boxes_per_class=100,
        pre_top_k=1,
        keep_top_k=1,
        background_label_id=-1),
    module=['mmyolo.deploy'])
# 设置代码库配置参数，包括类型、任务、模型类型、后处理参数、模块
```