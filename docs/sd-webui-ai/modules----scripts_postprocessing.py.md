# `stable-diffusion-webui\modules\scripts_postprocessing.py`

```py
# 导入必要的模块
import dataclasses
import os
import gradio as gr

# 导入自定义模块
from modules import errors, shared

# 定义一个数据类，用于存储后处理图像的共享信息
@dataclasses.dataclass
class PostprocessedImageSharedInfo:
    target_width: int = None
    target_height: int = None

# 定义后处理图像类
class PostprocessedImage:
    def __init__(self, image):
        # 初始化图像属性
        self.image = image
        self.info = {}
        self.shared = PostprocessedImageSharedInfo()
        self.extra_images = []
        self.nametags = []
        self.disable_processing = False
        self.caption = None

    # 获取后缀名
    def get_suffix(self, used_suffixes=None):
        used_suffixes = {} if used_suffixes is None else used_suffixes
        suffix = "-".join(self.nametags)
        if suffix:
            suffix = "-" + suffix

        if suffix not in used_suffixes:
            used_suffixes[suffix] = 1
            return suffix

        for i in range(1, 100):
            proposed_suffix = suffix + "-" + str(i)

            if proposed_suffix not in used_suffixes:
                used_suffixes[proposed_suffix] = 1
                return proposed_suffix

        return suffix

    # 创建图像副本
    def create_copy(self, new_image, *, nametags=None, disable_processing=False):
        pp = PostprocessedImage(new_image)
        pp.shared = self.shared
        pp.nametags = self.nametags.copy()
        pp.info = self.info.copy()
        pp.disable_processing = disable_processing

        if nametags is not None:
            pp.nametags += nametags

        return pp

# 脚本后处理类
class ScriptPostprocessing:
    filename = None
    controls = None
    args_from = None
    args_to = None

    order = 1000
    """scripts will be ordred by this value in postprocessing UI"""

    name = None
    """this function should return the title of the script."""

    group = None
    """A gr.Group component that has all script's UI inside it"""
    # 创建 Gradio 用户界面元素的函数
    def ui(self):
        """
        This function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be a dictionary that maps parameter names to components used in processing.
        Values of those components will be passed to process() function.
        """
        # 在这里编写创建用户界面元素的代码
        pass

    # 图像后处理函数，用于处理图像
    def process(self, pp: PostprocessedImage, **args):
        """
        This function is called to postprocess the image.
        args contains a dictionary with all values returned by components from ui()
        """
        # 在这里编写处理图像的代码，args 包含从 ui() 返回的所有值
        pass

    # 在调用 process() 函数之前为所有脚本调用的函数。脚本可以在这里检查图像并设置 pp 对象的字段以与其他脚本通信。
    def process_firstpass(self, pp: PostprocessedImage, **args):
        """
        Called for all scripts before calling process(). Scripts can examine the image here and set fields
        of the pp object to communicate things to other scripts.
        args contains a dictionary with all values returned by components from ui()
        """
        # 在这里编写在调用 process() 函数之前执行的代码，args 包含从 ui() 返回的所有值
        pass

    # 图像改变时调用的函数
    def image_changed(self):
        # 在这里编写图像改变时执行的代码
        pass
# 封装函数，调用指定函数并处理异常，返回结果或默认值
def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        # 调用指定函数并传入参数
        res = func(*args, **kwargs)
        # 返回函数调用结果
        return res
    except Exception as e:
        # 处理异常，显示错误信息
        errors.display(e, f"calling {filename}/{funcname}")
    
    # 返回默认值
    return default

# 定义脚本后处理运行器类
class ScriptPostprocessingRunner:
    def __init__(self):
        # 初始化脚本列表和 UI 创建状态
        self.scripts = None
        self.ui_created = False

    # 初始化脚本列表
    def initialize_scripts(self, scripts_data):
        # 清空脚本列表
        self.scripts = []

        # 遍历脚本数据，创建脚本对象并添加到脚本列表中
        for script_data in scripts_data:
            script: ScriptPostprocessing = script_data.script_class()
            script.filename = script_data.path

            # 如果脚本名为"Simple Upscale"，则跳过
            if script.name == "Simple Upscale":
                continue

            self.scripts.append(script)

    # 创建脚本的用户界面
    def create_script_ui(self, script, inputs):
        # 设置脚本参数范围
        script.args_from = len(inputs)
        script.args_to = len(inputs)

        # 调用函数创建脚本的用户界面控件
        script.controls = wrap_call(script.ui, script.filename, "ui")

        # 遍历脚本的控件，设置自定义脚本来源
        for control in script.controls.values():
            control.custom_script_source = os.path.basename(script.filename)

        # 将控件添加到输入列表中
        inputs += list(script.controls.values())
        script.args_to = len(inputs)

    # 按照优选顺序返回脚本列表
    def scripts_in_preferred_order(self):
        # 如果脚本列表为空，则初始化脚本列表
        if self.scripts is None:
            import modules.scripts
            self.initialize_scripts(modules.scripts.postprocessing_scripts_data)

        # 获取脚本操作顺序
        scripts_order = shared.opts.postprocessing_operation_order

        # 计算脚本得分
        def script_score(name):
            for i, possible_match in enumerate(scripts_order):
                if possible_match == name:
                    return i

            return len(self.scripts)

        # 构建脚本得分字典
        script_scores = {script.name: (script_score(script.name), script.order, script.name, original_index) for original_index, script in enumerate(self.scripts)}

        # 根据脚本得分排序脚本列表并返回
        return sorted(self.scripts, key=lambda x: script_scores[x.name])
    # 设置用户界面
    def setup_ui(self):
        # 初始化输入列表
        inputs = []

        # 遍历按照首选顺序排列的脚本
        for script in self.scripts_in_preferred_order():
            # 创建一个行组
            with gr.Row() as group:
                # 创建脚本的用户界面，将输入添加到列表中
                self.create_script_ui(script, inputs)

            # 将脚本与组关联
            script.group = group

        # 标记用户界面已创建
        self.ui_created = True
        # 返回输入列表
        return inputs

    # 运行处理
    def run(self, pp: PostprocessedImage, args):
        # 初始化脚本列表
        scripts = []

        # 遍历按照首选顺序排列的脚本
        for script in self.scripts_in_preferred_order():
            # 获取脚本的参数
            script_args = args[script.args_from:script.args_to]

            # 初始化处理参数字典
            process_args = {}
            # 将控件和参数值对应起来
            for (name, _component), value in zip(script.controls.items(), script_args):
                process_args[name] = value

            # 将脚本和处理参数添加到列表中
            scripts.append((script, process_args))

        # 遍历脚本和处理参数列表
        for script, process_args in scripts:
            # 对图像进行第一次处理
            script.process_firstpass(pp, **process_args)

        # 初始化所有图像列表，包含原始图像
        all_images = [pp]

        # 遍历脚本和处理参数列表
        for script, process_args in scripts:
            # 如果状态为跳过，则中断处理
            if shared.state.skipped:
                break

            # 设置当前作业为脚本名称
            shared.state.job = script.name

            # 遍历所有图像的副本
            for single_image in all_images.copy():

                # 如果图像未禁用处理，则对其进行处理
                if not single_image.disable_processing:
                    script.process(single_image, **process_args)

                # 遍历额外图像列表
                for extra_image in single_image.extra_images:
                    # 如果额外图像不是后处理图像，则创建其副本
                    if not isinstance(extra_image, PostprocessedImage):
                        extra_image = single_image.create_copy(extra_image)

                    # 将额外图像添加到所有图像列表中
                    all_images.append(extra_image)

                # 清空单个图像的额外图像列表
                single_image.extra_images.clear()

        # 将后处理图像的额外图像设置为除第一个图像外的所有图像
        pp.extra_images = all_images[1:]
    # 创建用于运行的参数列表
    def create_args_for_run(self, scripts_args):
        # 如果 UI 尚未创建，则使用禁用分析功能的 Blocks 上下文创建 UI
        if not self.ui_created:
            with gr.Blocks(analytics_enabled=False):
                self.setup_ui()

        # 按照首选顺序获取脚本列表
        scripts = self.scripts_in_preferred_order()
        # 创建一个包含最大参数数量的参数列表
        args = [None] * max([x.args_to for x in scripts])

        # 遍历脚本列表
        for script in scripts:
            # 获取当前脚本的参数字典
            script_args_dict = scripts_args.get(script.name, None)
            # 如果参数字典不为空
            if script_args_dict is not None:
                # 遍历当前脚本的控件列表
                for i, name in enumerate(script.controls):
                    # 将参数值添加到参数列表中
                    args[script.args_from + i] = script_args_dict.get(name, None)

        # 返回参数列表
        return args

    # 当图像改变时调用
    def image_changed(self):
        # 遍历按照首选顺序排列的脚本列表
        for script in self.scripts_in_preferred_order():
            # 调用每个脚本的图像改变方法
            script.image_changed()
```