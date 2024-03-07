# `.\YOLO-World\configs\deploy\detection_tensorrt-int8_static-640x640.py`

```
_base_ = [
    '../../third_party/mmdeploy/configs/mmdet/_base_/base_static.py',  # 定义基础配置文件路径
    '../../third_party/mmdeploy/configs/_base_/backends/tensorrt-int8.py']  # 定义后端配置文件路径

onnx_config = dict(input_shape=(640, 640))  # 定义输入形状配置

backend_config = dict(  # 定义后端配置
    common_config=dict(max_workspace_size=1 << 30),  # 定义通用配置，设置最大工作空间大小
    model_inputs=[  # 定义模型输入配置
        dict(  # 第一个模型输入配置
            input_shapes=dict(  # 定义输入形状
                input=dict(  # 输入名称为input
                    min_shape=[1, 3, 640, 640],  # 最小形状
                    opt_shape=[1, 3, 640, 640],  # 最佳形状
                    max_shape=[1, 3, 640, 640])))  # 最大形状
    ])

codebase_config = dict(  # 定义代码库配置
    type='mmyolo',  # 模型类型为mmyolo
    task='ObjectDetection',  # 任务为目标检测
    model_type='end2end',  # 模型类型为端到端
    post_processing=dict(  # 后处理配置
        score_threshold=0.1,  # 分数阈值
        confidence_threshold=0.005,  # 置信度阈值
        iou_threshold=0.3,  # IoU阈值
        max_output_boxes_per_class=100,  # 每类最大输出框数
        pre_top_k=1000,  # 预选框数
        keep_top_k=100,  # 保留框数
        background_label_id=-1),  # 背景标签ID
    module=['mmyolo.deploy'])  # 模块为mmyolo.deploy
```