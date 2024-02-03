# `stable-diffusion-webui\modules\gradio_extensons.py`

```
# 导入 gradio 库
import gradio as gr

# 从 modules 模块中导入 scripts, ui_tempdir, patches
from modules import scripts, ui_tempdir, patches

# 为组件添加样式类，用于 CSS 样式化
def add_classes_to_gradio_component(comp):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    """

    # 将组件的块名称添加到样式类中
    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(comp.elem_classes or [])]

    # 如果组件是多选的，添加 'multiselect' 样式类
    if getattr(comp, 'multiselect', False):
        comp.elem_classes.append('multiselect')

# 初始化 IOComponent
def IOComponent_init(self, *args, **kwargs):
    # 从参数中弹出 'tooltip' 键值对，并赋值给 self.webui_tooltip
    self.webui_tooltip = kwargs.pop('tooltip', None)

    # 如果当前脚本不为空，则在组件之前执行脚本
    if scripts.scripts_current is not None:
        scripts.scripts_current.before_component(self, **kwargs)

    # 在组件之前执行脚本回调
    scripts.script_callbacks.before_component_callback(self, **kwargs)

    # 调用原始的 IOComponent 初始化方法
    res = original_IOComponent_init(self, *args, **kwargs)

    # 为组件添加样式类
    add_classes_to_gradio_component(self)

    # 在组件之后执行脚本回调
    scripts.script_callbacks.after_component_callback(self, **kwargs)

    # 如果当前脚本不为空，则在组件之后执行脚本
    if scripts.scripts_current is not None:
        scripts.scripts_current.after_component(self, **kwargs)

    return res

# 获取块的配置信息
def Block_get_config(self):
    # 调用原始的 Block_get_config 方法获取配置信息
    config = original_Block_get_config(self)

    # 获取组件的 webui_tooltip 属性，如果存在则添加到配置中
    webui_tooltip = getattr(self, 'webui_tooltip', None)
    if webui_tooltip:
        config["webui_tooltip"] = webui_tooltip

    # 移除配置中的 'example_inputs' 键值对
    config.pop('example_inputs', None)

    return config

# 初始化 BlockContext
def BlockContext_init(self, *args, **kwargs):
    # 如果当前脚本不为空，则在组件之前执行脚本
    if scripts.scripts_current is not None:
        scripts.scripts_current.before_component(self, **kwargs)

    # 在组件之前执行脚本回调
    scripts.script_callbacks.before_component_callback(self, **kwargs)

    # 调用原始的 BlockContext 初始化方法
    res = original_BlockContext_init(self, *args, **kwargs)

    # 为组件添加样式类
    add_classes_to_gradio_component(self)

    # 在组件之后执行脚本回调
    scripts.script_callbacks.after_component_callback(self, **kwargs)

    # 如果当前脚本不为空，则在组件之后执行脚本
    if scripts.scripts_current is not None:
        scripts.scripts_current.after_component(self, **kwargs)

    return res

# 获取 Blocks 的配置文件
def Blocks_get_config_file(self, *args, **kwargs):
    # 调用原始的 Blocks_get_config_file 方法获取配置文件
    config = original_Blocks_get_config_file(self, *args, **kwargs)
    # 遍历配置文件中的组件列表
    for comp_config in config["components"]:
        # 检查组件配置中是否包含示例输入
        if "example_inputs" in comp_config:
            # 如果包含示例输入，则将其重置为空列表
            comp_config["example_inputs"] = {"serialized": []}

    # 返回更新后的配置文件
    return config
# 用于替换指定模块中指定类的指定方法，将原始方法替换为新方法
original_IOComponent_init = patches.patch(__name__, obj=gr.components.IOComponent, field="__init__", replacement=IOComponent_init)
original_Block_get_config = patches.patch(__name__, obj=gr.blocks.Block, field="get_config", replacement=Block_get_config)
original_BlockContext_init = patches.patch(__name__, obj=gr.blocks.BlockContext, field="__init__", replacement=BlockContext_init)
original_Blocks_get_config_file = patches.patch(__name__, obj=gr.blocks.Blocks, field="get_config_file", replacement=Blocks_get_config_file)

# 安装 UI 临时目录覆盖
ui_tempdir.install_ui_tempdir_override()
```