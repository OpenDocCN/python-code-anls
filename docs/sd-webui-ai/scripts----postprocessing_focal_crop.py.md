# `stable-diffusion-webui\scripts\postprocessing_focal_crop.py`

```py
# 从 modules 模块中导入 scripts_postprocessing、ui_components、errors 模块
from modules import scripts_postprocessing, ui_components, errors
# 导入 gradio 库并重命名为 gr
import gradio as gr

# 创建 ScriptPostprocessingFocalCrop 类，继承自 scripts_postprocessing.ScriptPostprocessing 类
class ScriptPostprocessingFocalCrop(scripts_postprocessing.ScriptPostprocessing):
    # 定义类属性 name 为 "Auto focal point crop"
    name = "Auto focal point crop"
    # 定义类属性 order 为 4000
    order = 4000

    # 定义 ui 方法
    def ui(self):
        # 创建一个 InputAccordion 对象，label 为 "Auto focal point crop"，默认展开状态为 False
        with ui_components.InputAccordion(False, label="Auto focal point crop") as enable:
            # 创建一个滑块组件，label 为 'Focal point face weight'，默认值为 0.9，最小值为 0.0，最大值为 1.0，步长为 0.05，元素 ID 为 "postprocess_focal_crop_face_weight"
            face_weight = gr.Slider(label='Focal point face weight', value=0.9, minimum=0.0, maximum=1.0, step=0.05, elem_id="postprocess_focal_crop_face_weight")
            # 创建一个滑块组件，label 为 'Focal point entropy weight'，默认值为 0.15，最小值为 0.0，最大值为 1.0，步长为 0.05，元素 ID 为 "postprocess_focal_crop_entropy_weight"
            entropy_weight = gr.Slider(label='Focal point entropy weight', value=0.15, minimum=0.0, maximum=1.0, step=0.05, elem_id="postprocess_focal_crop_entropy_weight")
            # 创建一个滑块组件，label 为 'Focal point edges weight'，默认值为 0.5，最小值为 0.0，最大值为 1.0，步长为 0.05，元素 ID 为 "postprocess_focal_crop_edges_weight"
            edges_weight = gr.Slider(label='Focal point edges weight', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="postprocess_focal_crop_edges_weight")
            # 创建一个复选框组件，label 为 'Create debug image'，元素 ID 为 "train_process_focal_crop_debug"
            debug = gr.Checkbox(label='Create debug image', elem_id="train_process_focal_crop_debug")

        # 返回一个字典，包含 "enable"、"face_weight"、"entropy_weight"、"edges_weight"、"debug" 键值对
        return {
            "enable": enable,
            "face_weight": face_weight,
            "entropy_weight": entropy_weight,
            "edges_weight": edges_weight,
            "debug": debug,
        }
    # 处理后处理图像的方法，根据参数进行自动裁剪
    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, face_weight, entropy_weight, edges_weight, debug):
        # 如果未启用自动裁剪功能，则直接返回
        if not enable:
            return

        # 如果目标宽度或目标高度未设置，则直接返回
        if not pp.shared.target_width or not pp.shared.target_height:
            return

        # 初始化 DNN 模型路径为 None
        dnn_model_path = None
        try:
            # 尝试下载并缓存人脸检测模型
            dnn_model_path = autocrop.download_and_cache_models()
        except Exception:
            # 如果下载模型失败，则报告错误并使用较低质量的 haar 方法作为备选
            errors.report("Unable to load face detection model for auto crop selection. Falling back to lower quality haar method.", exc_info=True)

        # 设置自动裁剪的参数
        autocrop_settings = autocrop.Settings(
            crop_width=pp.shared.target_width,
            crop_height=pp.shared.target_height,
            face_points_weight=face_weight,
            entropy_points_weight=entropy_weight,
            corner_points_weight=edges_weight,
            annotate_image=debug,
            dnn_model_path=dnn_model_path,
        )

        # 调用自动裁剪方法，获取裁剪后的结果图像和其他图像
        result, *others = autocrop.crop_image(pp.image, autocrop_settings)

        # 将处理后的结果图像赋值给后处理图像对象
        pp.image = result
        # 创建其他图像的副本，并添加到额外图像列表中
        pp.extra_images = [pp.create_copy(x, nametags=["focal-crop-debug"], disable_processing=True) for x in others]
```