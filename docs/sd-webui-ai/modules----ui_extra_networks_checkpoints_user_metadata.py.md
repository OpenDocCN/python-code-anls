# `stable-diffusion-webui\modules\ui_extra_networks_checkpoints_user_metadata.py`

```
# 导入 gradio 库
import gradio as gr

# 从 modules 模块中导入 ui_extra_networks_user_metadata、sd_vae、shared 模块
from modules import ui_extra_networks_user_metadata, sd_vae, shared
# 从 modules 模块中导入 ui_common 模块中的 create_refresh_button 函数

# 定义 CheckpointUserMetadataEditor 类，继承自 ui_extra_networks_user_metadata.UserMetadataEditor 类
class CheckpointUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
    # 初始化方法
    def __init__(self, ui, tabname, page):
        # 调用父类的初始化方法
        super().__init__(ui, tabname, page)

        # 初始化 select_vae 属性为 None

    # 保存用户元数据的方法
    def save_user_metadata(self, name, desc, notes, vae):
        # 获取指定名称的用户元数据
        user_metadata = self.get_user_metadata(name)
        # 更新用户元数据的描述、注释和 vae 属性
        user_metadata["description"] = desc
        user_metadata["notes"] = notes
        user_metadata["vae"] = vae

        # 写入更新后的用户元数据
        self.write_user_metadata(name, user_metadata)

    # 更新 vae 的方法
    def update_vae(self, name):
        # 如果名称与 shared.sd_model.sd_checkpoint_info.name_for_extra 相同
        if name == shared.sd_model.sd_checkpoint_info.name_for_extra:
            # 重新加载 vae 权重
            sd_vae.reload_vae_weights()

    # 将值放入组件中的方法
    def put_values_into_components(self, name):
        # 获取指定名称的用户元数据
        user_metadata = self.get_user_metadata(name)
        # 调用父类的 put_values_into_components 方法，并获取返回值
        values = super().put_values_into_components(name)

        # 返回更新后的值列表，包括前5个值和用户元数据中的 vae 值
        return [
            *values[0:5],
            user_metadata.get('vae', ''),
        ]
    # 创建编辑器界面
    def create_editor(self):
        # 创建默认的编辑器元素
        self.create_default_editor_elems()

        # 创建一个行布局
        with gr.Row():
            # 创建下拉框，用于选择首选的 VAE 模型
            self.select_vae = gr.Dropdown(choices=["Automatic", "None"] + list(sd_vae.vae_dict), value="None", label="Preferred VAE", elem_id="checpoint_edit_user_metadata_preferred_vae")
            # 创建刷新按钮，用于刷新 VAE 列表
            create_refresh_button(self.select_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["Automatic", "None"] + list(sd_vae.vae_dict)}, "checpoint_edit_user_metadata_refresh_preferred_vae")

        # 创建文本区域，用于输入备注信息
        self.edit_notes = gr.TextArea(label='Notes', lines=4)

        # 创建默认的按钮
        self.create_default_buttons()

        # 将需要显示的组件放入列表
        viewed_components = [
            self.edit_name,
            self.edit_description,
            self.html_filedata,
            self.html_preview,
            self.edit_notes,
            self.select_vae,
        ]

        # 点击编辑按钮时，将值放入组件中
        self.button_edit\
            .click(fn=self.put_values_into_components, inputs=[self.edit_name_input], outputs=viewed_components)\
            .then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[self.box])

        # 将需要编辑的组件放入列表
        edited_components = [
            self.edit_description,
            self.edit_notes,
            self.select_vae,
        ]

        # 设置保存处理程序，保存用户元数据
        self.setup_save_handler(self.button_save, self.save_user_metadata, edited_components)
        # 点击保存按钮时，更新 VAE
        self.button_save.click(fn=self.update_vae, inputs=[self.edit_name_input])
```