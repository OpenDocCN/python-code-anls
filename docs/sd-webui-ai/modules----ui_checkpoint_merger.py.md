# `stable-diffusion-webui\modules\ui_checkpoint_merger.py`

```py
# 导入 gradio 库
import gradio as gr

# 从 modules 模块中导入 sd_models, sd_vae, errors, extras, call_queue
from modules import sd_models, sd_vae, errors, extras, call_queue
# 从 modules.ui_components 模块中导入 FormRow
from modules.ui_components import FormRow
# 从 modules.ui_common 模块中导入 create_refresh_button

# 更新插值描述
def update_interp_description(value):
    # 插值描述的 CSS 样式
    interp_description_css = "<p style='margin-bottom: 2.5em'>{}</p>"
    # 插值描述字典
    interp_descriptions = {
        "No interpolation": interp_description_css.format("No interpolation will be used. Requires one model; A. Allows for format conversion and VAE baking."),
        "Weighted sum": interp_description_css.format("A weighted sum will be used for interpolation. Requires two models; A and B. The result is calculated as A * (1 - M) + B * M"),
        "Add difference": interp_description_css.format("The difference between the last two models will be added to the first. Requires three models; A, B and C. The result is calculated as A + (B - C) * M")
    }
    # 返回对应值的插值描述
    return interp_descriptions[value]

# 模型合并函数
def modelmerger(*args):
    try:
        # 运行模型合并函数
        results = extras.run_modelmerger(*args)
    except Exception as e:
        # 报告错误信息
        errors.report("Error loading/saving model file", exc_info=True)
        # 列出模型列表，以移除可能缺失的模型
        sd_models.list_models()
        # 返回错误信息
        return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], f"Error merging checkpoints: {e}"]
    # 返回结果
    return results

# 定义 UiCheckpointMerger 类
class UiCheckpointMerger:
    # 设置用户界面，根据选择的检查点格式改变可见性
    self.checkpoint_format.change(lambda fmt: gr.update(visible=fmt == 'safetensors'), inputs=[self.checkpoint_format], outputs=[self.metadata_editor], show_progress=False)

    # 点击按钮读取元数据，输入为主模型名称、次要模型名称和第三模型名称，输出为元数据 JSON
    self.read_metadata.click(extras.read_metadata, inputs=[self.primary_model_name, self.secondary_model_name, self.tertiary_model_name], outputs=[self.metadata_json])

    # 点击按钮执行模型合并，无输入，输出为模型合并结果
    self.modelmerger_merge.click(fn=lambda: '', inputs=[], outputs=[self.modelmerger_result])
    # 点击按钮执行模型合并，调用 GPU 加速的 modelmerger 函数，设置输入和输出
    self.modelmerger_merge.click(
        fn=call_queue.wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [gr.update() for _ in range(4)]),
        _js='modelmerger',
        inputs=[
            dummy_component,
            self.primary_model_name,
            self.secondary_model_name,
            self.tertiary_model_name,
            self.interp_method,
            self.interp_amount,
            self.save_as_half,
            self.custom_name,
            self.checkpoint_format,
            self.config_source,
            self.bake_in_vae,
            self.discard_weights,
            self.save_metadata,
            self.add_merge_recipe,
            self.copy_metadata_fields,
            self.metadata_json,
        ],
        outputs=[
            self.primary_model_name,
            self.secondary_model_name,
            self.tertiary_model_name,
            sd_model_checkpoint_component,
            self.modelmerger_result,
        ]
    )

    # 为了解决从 ui-config.json 加载值时 change() 事件不触发的问题，更新插值描述的值
    self.interp_description.value = update_interp_description(self.interp_method.value)
```