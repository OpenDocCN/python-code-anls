# `stable-diffusion-webui\modules\scripts.py`

```py
# 导入必要的模块
import os
import re
import sys
import inspect
from collections import namedtuple
from dataclasses import dataclass

# 导入 gradio 模块
import gradio as gr

# 导入自定义模块
from modules import shared, paths, script_callbacks, extensions, script_loading, scripts_postprocessing, errors, timer

# 定义一个常量对象 AlwaysVisible
AlwaysVisible = object()

# 定义一个类 PostprocessImageArgs，用于处理单个图像的后处理参数
class PostprocessImageArgs:
    def __init__(self, image):
        self.image = image

# 定义一个类 PostprocessBatchListArgs，用于处理批量图像的后处理参数
class PostprocessBatchListArgs:
    def __init__(self, images):
        self.images = images

# 使用 dataclass 装饰器定义一个数据类 OnComponent，用于表示组件
@dataclass
class OnComponent:
    component: gr.blocks.Block

# 定义一个脚本类 Script，用于表示脚本的相关信息
class Script:
    name = None
    """script's internal name derived from title"""

    section = None
    """name of UI section that the script's controls will be placed into"""

    filename = None
    args_from = None
    args_to = None
    alwayson = False

    is_txt2img = False
    is_img2img = False
    tabname = None

    group = None
    """A gr.Group component that has all script's UI inside it."""

    create_group = True
    """If False, for alwayson scripts, a group component will not be created."""

    infotext_fields = None
    """if set in ui(), this is a list of pairs of gradio component + text; the text will be used when
    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example
    """

    paste_field_names = None
    """if set in ui(), this is a list of names of infotext fields; the fields will be sent through the
    various "Send to <X>" buttons when clicked
    """

    api_info = None
    """Generated value of type modules.api.models.ScriptInfo with information about the script for API"""

    on_before_component_elem_id = None
    """list of callbacks to be called before a component with an elem_id is created"""

    on_after_component_elem_id = None
    """list of callbacks to be called after a component with an elem_id is created"""

    setup_for_ui_only = False
    """If true, the script setup will only be run in Gradio UI, not in API"""
    # 定义一个方法，用于返回脚本的标题。这将显示在下拉菜单中。
    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""
        raise NotImplementedError()

    # 定义一个方法，用于创建 Gradio UI 元素。参考 https://gradio.app/docs/#components
    # 返回值应该是一个包含所有用于处理的组件的数组。
    # 这些返回组件的值将传递给 run() 和 process() 函数。
    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        pass

    # 定义一个方法，用于显示脚本。如果是为 img2img 接口调用，则 is_img2img 为 True，否则为 False。
    # 返回值应该是：
    # - 如果脚本根本不应该在 UI 中显示，则返回 False
    # - 如果脚本应该在 UI 中显示，但只有在脚本下拉菜单中选择时才显示，则返回 True
    # - 如果脚本应该始终在 UI 中显示，则返回 script.AlwaysVisible
    def show(self, is_img2img):
        """
        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
         """
        return True

    # 定义一个方法，用于运行脚本。如果脚本在脚本下拉菜单中被选择，则调用此函数。
    # 必须进行所有处理并返回具有结果的 Processed 对象，与 processing.process_images 返回的对象相同。
    # 通常通过调用 processing.process_images 函数来进行处理。
    # args 包含来自 ui() 函数返回的所有组件的值。
    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.

        Usually the processing is done by calling the processing.process_images function.

        args contains all values returned by components from ui()
        """
        pass

    # 对于始终可见的脚本，设置处理对象之前调用此函数，在任何处理开始之前。
    # args 包含来自 ui() 函数返回的所有组件的值。
    def setup(self, p, *args):
        """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
        args contains all values returned by components from ui().
        """
        pass

    # 对于始终可见的脚本，在处理开始时非常早调用此函数。
    # 可以在这里修改处理对象 (p)，注入钩子等。
    # args 包含来自 ui() 函数返回的所有组件的值。
    def before_process(self, p, *args):
        """
        This function is called very early during processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        pass
    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def after_extra_networks_activate(self, p, *args, **kwargs):
        """
        Called after extra networks activation, before conds calculation
        allow modification of the network after extra networks activation been applied
        won't be call if p.disable_extra_networks

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
          - extra_network_data - list of ExtraNetworkParams for current stage
        """
        pass
    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """

        pass

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, *args, **kwargs):
        """
        Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.
        This is useful when you want to update the entire batch instead of individual images.

        You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.
        If the number of images is different from the batch size when returning,
        then the script has the responsibility to also update the following attributes in the processing object (p):
          - p.prompts
          - p.negative_prompts
          - p.seeds
          - p.subseeds

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
        """

        pass
    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """

        pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """

        pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """

        pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """

        pass

    def on_before_component(self, callback, *, elem_id):
        """
        Calls callback before a component is created. The callback function is called with a single argument of type OnComponent.

        May be called in show() or ui() - but it may be too late in latter as some components may already be created.

        This function is an alternative to before_component in that it also cllows to run before a component is created, but
        it doesn't require to be called for every created component - just for the one you need.
        """
        if self.on_before_component_elem_id is None:
            self.on_before_component_elem_id = []

        self.on_before_component_elem_id.append((elem_id, callback))
    # 在组件创建后调用回调函数，回调函数接受一个 OnComponent 类型的参数
    def on_after_component(self, callback, *, elem_id):
        """
        Calls callback after a component is created. The callback function is called with a single argument of type OnComponent.
        """
        # 如果 on_after_component_elem_id 为空，则初始化为一个空列表
        if self.on_after_component_elem_id is None:
            self.on_after_component_elem_id = []

        # 将元素 ID 和回调函数添加到 on_after_component_elem_id 列表中
        self.on_after_component_elem_id.append((elem_id, callback))

    # 描述函数，未使用
    def describe(self):
        """unused"""
        # 返回空字符串
        return ""

    # 生成 HTML 元素的 ID 的辅助函数，构建最终的 ID 由脚本名称、标签和用户提供的 item_id 组成
    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""

        # 检查是否需要标签名称
        need_tabname = self.show(True) == self.show(False)
        tabkind = 'img2img' if self.is_img2img else 'txt2img'
        tabname = f"{tabkind}_" if need_tabname else ""
        # 将标题转换为小写，并替换非字母数字字符和空格为下划线
        title = re.sub(r'[^a-z_0-9]', '', re.sub(r'\s', '_', self.title().lower()))

        # 返回生成的元素 ID
        return f'script_{tabname}{title}_{item_id}'

    # 在 hires 修复开始之前调用的函数
    def before_hr(self, p, *args):
        """
        This function is called before hires fix start.
        """
        # 空函数，不执行任何操作
        pass
# 定义一个名为ScriptBuiltinUI的类，继承自Script类，用于处理内置UI相关的脚本
class ScriptBuiltinUI(Script):
    # 标记该类仅用于UI设置
    setup_for_ui_only = True

    # 定义一个名为elem_id的方法，用于生成HTML元素的id，根据tab和用户提供的item_id构建最终id
    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of tab and user-supplied item_id"""

        # 检查是否需要tab名称
        need_tabname = self.show(True) == self.show(False)
        # 根据需要的tab名称和是否为img2img类型生成tab名称
        tabname = ('img2img' if self.is_img2img else 'txt2img') + "_" if need_tabname else ""

        # 返回构建好的id
        return f'{tabname}{item_id}'


# 定义当前脚本的基本目录为paths.script_path
current_basedir = paths.script_path


# 定义一个名为basedir的方法，返回当前脚本的基本目录
def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


# 定义一个名为ScriptFile的命名元组，包含basedir、filename和path三个字段
ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])

# 初始化脚本数据列表和后处理脚本数据列表
scripts_data = []
postprocessing_scripts_data = []

# 定义一个名为ScriptClassData的命名元组，包含script_class、path、basedir和module四个字段
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])

# 定义一个名为topological_sort的方法，用于对依赖关系进行拓扑排序
def topological_sort(dependencies):
    """Accepts a dictionary mapping name to its dependencies, returns a list of names ordered according to dependencies.
    Ignores errors relating to missing dependeencies or circular dependencies
    """

    # 初始化已访问字典和结果列表
    visited = {}
    result = []

    # 定义内部递归方法inner，用于深度优先搜索
    def inner(name):
        visited[name] = True

        # 遍历当前节点的依赖
        for dep in dependencies.get(name, []):
            if dep in dependencies and dep not in visited:
                inner(dep)

        result.append(name)

    # 遍历所有依赖关系，进行拓扑排序
    for depname in dependencies:
        if depname not in visited:
            inner(depname)

    return result


# 定义一个名为ScriptWithDependencies的数据类，包含script_canonical_name、file、requires、load_before和load_after字段
@dataclass
class ScriptWithDependencies:
    script_canonical_name: str
    file: ScriptFile
    requires: list
    load_before: list
    load_after: list


# 定义一个名为list_scripts的方法，用于列出脚本
def list_scripts(scriptdirname, extension, *, include_extensions=True):
    # 初始化脚本字典
    scripts = {}

    # 获取已加载的扩展
    loaded_extensions = {ext.canonical_name: ext for ext in extensions.active()}
    # 创建一个空字典，用于存储已加载的扩展脚本
    loaded_extensions_scripts = {ext.canonical_name: [] for ext in extensions.active()}

    # 构建脚本依赖关系图
    root_script_basedir = os.path.join(paths.script_path, scriptdirname)
    # 如果根目录存在
    if os.path.exists(root_script_basedir):
        # 遍历根目录下的文件
        for filename in sorted(os.listdir(root_script_basedir)):
            # 如果不是文件，则跳过
            if not os.path.isfile(os.path.join(root_script_basedir, filename)):
                continue

            # 如果文件扩展名不匹配指定的扩展名，则跳过
            if os.path.splitext(filename)[1].lower() != extension:
                continue

            # 创建脚本文件对象
            script_file = ScriptFile(paths.script_path, filename, os.path.join(root_script_basedir, filename))
            # 创建脚本对象并添加到字典中
            scripts[filename] = ScriptWithDependencies(filename, script_file, [], [], [])

    # 如果需要包含扩展
    if include_extensions:
        # 遍历活动的扩展
        for ext in extensions.active():
            # 获取扩展中与指定目录和扩展名匹配的脚本列表
            extension_scripts_list = ext.list_files(scriptdirname, extension)
            # 遍历扩展脚本列表
            for extension_script in extension_scripts_list:
                # 如果不是文件，则跳过
                if not os.path.isfile(extension_script.path):
                    continue

                # 构建脚本的规范名称
                script_canonical_name = ("builtin/" if ext.is_builtin else "") + ext.canonical_name + "/" + extension_script.filename
                relative_path = scriptdirname + "/" + extension_script.filename

                # 创建脚本对象并添加到字典中
                script = ScriptWithDependencies(
                    script_canonical_name=script_canonical_name,
                    file=extension_script,
                    requires=ext.metadata.get_script_requirements("Requires", relative_path, scriptdirname),
                    load_before=ext.metadata.get_script_requirements("Before", relative_path, scriptdirname),
                    load_after=ext.metadata.get_script_requirements("After", relative_path, scriptdirname),
                )

                scripts[script_canonical_name] = script
                # 将脚本添加到已加载的扩展脚本列表中
                loaded_extensions_scripts[ext.canonical_name].append(script)
    # 遍历所有脚本的规范名称和脚本对象
    for script_canonical_name, script in scripts.items():
        # 在加载之前需要反向依赖
        # 在这种情况下，将脚本名称添加到指定脚本的load_after列表中
        for load_before in script.load_before:
            # 如果需要在加载之前加载单个脚本
            other_script = scripts.get(load_before)
            if other_script:
                other_script.load_after.append(script_canonical_name)

            # 如果需要加载扩展
            other_extension_scripts = loaded_extensions_scripts.get(load_before)
            if other_extension_scripts:
                for other_script in other_extension_scripts:
                    other_script.load_after.append(script_canonical_name)

        # 如果After提到了一个扩展，删除它，而是添加其所有脚本
        for load_after in list(script.load_after):
            if load_after not in scripts and load_after in loaded_extensions_scripts:
                script.load_after.remove(load_after)

                for other_script in loaded_extensions_scripts.get(load_after, []):
                    script.load_after.append(other_script.script_canonical_name)

    # 初始化依赖关系字典
    dependencies = {}

    # 遍历所有脚本的规范名称和脚本对象
    for script_canonical_name, script in scripts.items():
        # 遍历脚本所需的所有脚本
        for required_script in script.requires:
            # 如果所需脚本既不在脚本中也不在加载的扩展中，则报告错误
            if required_script not in scripts and required_script not in loaded_extensions:
                errors.report(f'Script "{script_canonical_name}" requires "{required_script}" to be loaded, but it is not.', exc_info=False)

        # 将脚本的load_after列表作为依赖关系字典的值
        dependencies[script_canonical_name] = script.load_after

    # 对依赖关系进行拓扑排序
    ordered_scripts = topological_sort(dependencies)
    # 获取排序后的脚本文件列表
    scripts_list = [scripts[script_canonical_name].file for script_canonical_name in ordered_scripts]

    # 返回排序后的脚本文件列表
    return scripts_list
# 根据文件名列出所有包含该文件名的文件路径
def list_files_with_name(filename):
    # 初始化结果列表
    res = []

    # 获取所有目录，包括脚本路径和激活的扩展路径
    dirs = [paths.script_path] + [ext.path for ext in extensions.active()]

    # 遍历每个目录
    for dirpath in dirs:
        # 如果目录不存在，则跳过
        if not os.path.isdir(dirpath):
            continue

        # 拼接目录和文件名，得到文件路径
        path = os.path.join(dirpath, filename)
        # 如果该路径是文件，则将其添加到结果列表中
        if os.path.isfile(path):
            res.append(path)

    # 返回结果列表
    return res


# 加载脚本
def load_scripts():
    # 声明全局变量
    global current_basedir
    # 清空脚本数据、后处理脚本数据和脚本回调
    scripts_data.clear()
    postprocessing_scripts_data.clear()
    script_callbacks.clear_callbacks()

    # 获取所有脚本列表，包括处理脚本和后处理脚本
    scripts_list = list_scripts("scripts", ".py") + list_scripts("modules/processing_scripts", ".py", include_extensions=False)

    # 保存当前系统路径
    syspath = sys.path

    # 注册模块中的脚本
    def register_scripts_from_module(module):
        # 遍历模块中的所有类
        for script_class in module.__dict__.values():
            # 如果不是类，则跳过
            if not inspect.isclass(script_class):
                continue

            # 如果是 Script 类的子类，则将其添加到脚本数据中
            if issubclass(script_class, Script):
                scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))
            # 如果是 scripts_postprocessing.ScriptPostprocessing 类的子类，则将其添加到后处理脚本数据中
            elif issubclass(script_class, scripts_postprocessing.ScriptPostprocessing):
                postprocessing_scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))

    # 遍历脚本列表
    # 这里脚本列表已经是有序的
    # 不考虑处理脚本
    for scriptfile in scripts_list:
        try:
            # 如果脚本所在目录不是脚本路径，则将其添加到系统路径中
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir

            # 加载脚本模块
            script_module = script_loading.load_module(scriptfile.path)
            # 注册模块中的脚本
            register_scripts_from_module(script_module)

        except Exception:
            # 报告加载脚本时的错误
            errors.report(f"Error loading script: {scriptfile.filename}", exc_info=True)

        finally:
            # 恢复系统路径和当前基本目录
            sys.path = syspath
            current_basedir = paths.script_path
            # 记录脚本加载时间
            timer.startup_timer.record(scriptfile.filename)

    # 声明全局变量
    global scripts_txt2img, scripts_img2img, scripts_postproc
    # 创建一个脚本运行器对象用于将文本转换为图像
    scripts_txt2img = ScriptRunner()
    # 创建一个脚本运行器对象用于将图像转换为图像
    scripts_img2img = ScriptRunner()
    # 创建一个脚本后处理运行器对象
    scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()
# 定义一个函数，用于调用指定函数并处理异常
def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        # 调用指定函数并传入参数
        return func(*args, **kwargs)
    except Exception:
        # 报告调用函数时的错误信息
        errors.report(f"Error calling: {filename}/{funcname}", exc_info=True)
    
    # 返回默认值
    return default

# 定义一个脚本运行器类
class ScriptRunner:
    def __init__(self):
        # 初始化脚本列表、可选择脚本列表、始终运行脚本列表、标题列表、标题映射、信息文本字段列表、粘贴字段名称列表、输入列表
        self.scripts = []
        self.selectable_scripts = []
        self.alwayson_scripts = []
        self.titles = []
        self.title_map = {}
        self.infotext_fields = []
        self.paste_field_names = []
        self.inputs = [None]

        # 在创建组件元素之前的回调函数字典，键为元素ID，值为回调函数列表
        self.on_before_component_elem_id = {}

        # 在创建组件元素之后的回调函数字典，键为元素ID，值为回调函数列表
        self.on_after_component_elem_id = {}

    # 初始化脚本
    def initialize_scripts(self, is_img2img):
        # 导入模块中的自动后处理脚本
        from modules import scripts_auto_postprocessing

        # 清空脚本列表、始终运行脚本列表、可选择脚本列表
        self.scripts.clear()
        self.alwayson_scripts.clear()
        self.selectable_scripts.clear()

        # 创建自动预处理脚本数据
        auto_processing_scripts = scripts_auto_postprocessing.create_auto_preprocessing_script_data()

        # 遍历自动预处理脚本数据和脚本数据
        for script_data in auto_processing_scripts + scripts_data:
            # 创建脚本实例
            script = script_data.script_class()
            script.filename = script_data.path
            script.is_txt2img = not is_img2img
            script.is_img2img = is_img2img
            script.tabname = "img2img" if is_img2img else "txt2img"

            # 判断脚本是否可见
            visibility = script.show(script.is_img2img)

            # 如果始终可见
            if visibility == AlwaysVisible:
                self.scripts.append(script)
                self.alwayson_scripts.append(script)
                script.alwayson = True

            # 如果可见
            elif visibility:
                self.scripts.append(script)
                self.selectable_scripts.append(script)

        # 应用在创建组件之前的回调函数
        self.apply_on_before_component_callbacks()
    # 应用在组件之前的回调函数
    def apply_on_before_component_callbacks(self):
        # 遍历脚本列表
        for script in self.scripts:
            # 获取脚本中定义的在组件之前执行的回调函数列表
            on_before = script.on_before_component_elem_id or []
            # 获取脚本中定义的在组件之后执行的回调函数列表
            on_after = script.on_after_component_elem_id or []

            # 遍历在组件之前执行的回调函数列表
            for elem_id, callback in on_before:
                # 如果元素 ID 不在已有的回调函数字典中，则创建一个空列表
                if elem_id not in self.on_before_component_elem_id:
                    self.on_before_component_elem_id[elem_id] = []

                # 将回调函数和脚本添加到对应元素 ID 的回调函数列表中
                self.on_before_component_elem_id[elem_id].append((callback, script))

            # 遍历在组件之后执行的回调函数列表
            for elem_id, callback in on_after:
                # 如果元素 ID 不在已有的回调函数字典中，则创建一个空列表
                if elem_id not in self.on_after_component_elem_id:
                    self.on_after_component_elem_id[elem_id] = []

                # 将回调函数和脚本添加到对应元素 ID 的回调函数列表中
                self.on_after_component_elem_id[elem_id].append((callback, script))

            # 清空在组件之前和之后执行的回调函数列表
            on_before.clear()
            on_after.clear()

    # 创建脚本的用户界面
    def create_script_ui(self, script):
        # 设置脚本的参数起始和结束索引
        script.args_from = len(self.inputs)
        script.args_to = len(self.inputs)

        # 尝试创建脚本的用户界面，如果出现异常则报告错误信息
        try:
            self.create_script_ui_inner(script)
        except Exception:
            errors.report(f"Error creating UI for {script.name}: ", exc_info=True)
    # 创建脚本的用户界面，传入脚本对象
    def create_script_ui_inner(self, script):
        # 导入模块 modules.api.models 到当前命名空间
        import modules.api.models as api_models

        # 调用 wrap_call 函数，获取脚本的控件
        controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)

        # 如果控件为空，则返回
        if controls is None:
            return

        # 设置脚本的名称为脚本标题或默认为脚本文件名的小写形式
        script.name = wrap_call(script.title, script.filename, "title", default=script.filename).lower()

        # 初始化 API 参数列表
        api_args = []

        # 遍历控件列表
        for control in controls:
            # 设置控件的 custom_script_source 属性为脚本文件名的基本名称
            control.custom_script_source = os.path.basename(script.filename)

            # 创建 API 参数信息对象
            arg_info = api_models.ScriptArg(label=control.label or "")

            # 遍历字段列表，设置参数信息对象的字段值
            for field in ("value", "minimum", "maximum", "step"):
                v = getattr(control, field, None)
                if v is not None:
                    setattr(arg_info, field, v)

            # 获取控件的选择项列表，处理 gradio 3.41 版本中的选择项格式问题
            choices = getattr(control, 'choices', None)
            if choices is not None:
                arg_info.choices = [x[0] if isinstance(x, tuple) else x for x in choices]

            # 将参数信息对象添加到 API 参数列表中
            api_args.append(arg_info)

        # 设置脚本的 API 信息对象
        script.api_info = api_models.ScriptInfo(
            name=script.name,
            is_img2img=script.is_img2img,
            is_alwayson=script.alwayson,
            args=api_args,
        )

        # 如果脚本的 infotext_fields 不为空，则将其添加到当前对象的 infotext_fields 列表中
        if script.infotext_fields is not None:
            self.infotext_fields += script.infotext_fields

        # 如果脚本的 paste_field_names 不为空，则将其添加到当前对象的 paste_field_names 列表中
        if script.paste_field_names is not None:
            self.paste_field_names += script.paste_field_names

        # 将控件列表添加到当前对象的 inputs 列表中
        self.inputs += controls
        # 设置脚本的 args_to 属性为当前 inputs 列表的长度
        script.args_to = len(self.inputs)
    # 为特定部分设置用户界面，根据传入的脚本列表，默认为self.alwayson_scripts
    def setup_ui_for_section(self, section, scriptlist=None):
        # 如果脚本列表为空，则使用默认的self.alwayson_scripts
        if scriptlist is None:
            scriptlist = self.alwayson_scripts

        # 遍历脚本列表
        for script in scriptlist:
            # 如果脚本是始终运行的并且不属于当前部分，则跳过
            if script.alwayson and script.section != section:
                continue

            # 如果脚本需要创建组
            if script.create_group:
                # 创建一个可见性为script.alwayson的组
                with gr.Group(visible=script.alwayson) as group:
                    # 创建脚本的用户界面
                    self.create_script_ui(script)

                # 将组赋值给脚本的group属性
                script.group = group
            else:
                # 创建脚本的用户界面
                self.create_script_ui(script)

    # 准备用户界面
    def prepare_ui(self):
        # 初始化输入列表，包含一个None元素
        self.inputs = [None]

    # 运行脚本
    def run(self, p, *args):
        # 获取脚本索引
        script_index = args[0]

        # 如果脚本索引为0，则返回None
        if script_index == 0:
            return None

        # 获取选中的脚本
        script = self.selectable_scripts[script_index-1]

        # 如果选中的脚本为空，则返回None
        if script is None:
            return None

        # 获取脚本参数
        script_args = args[script.args_from:script.args_to]
        # 运行脚本并返回处理结果
        processed = script.run(p, *script_args)

        # 清空共享的total_tqdm
        shared.total_tqdm.clear()

        return processed

    # 在处理之前执行操作
    def before_process(self, p):
        # 遍历始终运行的脚本列表
        for script in self.alwayson_scripts:
            try:
                # 获取脚本参数并执行before_process方法
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_process(p, *script_args)
            except Exception:
                # 报告执行before_process时的错误信息
                errors.report(f"Error running before_process: {script.filename}", exc_info=True)

    # 处理操作
    def process(self, p):
        # 遍历始终运行的脚本列表
        for script in self.alwayson_scripts:
            try:
                # 获取脚本参数并执行process方法
                script_args = p.script_args[script.args_from:script.args_to]
                script.process(p, *script_args)
            except Exception:
                # 报告执行process时的错误信息
                errors.report(f"Error running process: {script.filename}", exc_info=True)
    # 在处理批次之前运行脚本
    def before_process_batch(self, p, **kwargs):
        # 遍历所有始终运行的脚本
        for script in self.alwayson_scripts:
            try:
                # 获取脚本参数
                script_args = p.script_args[script.args_from:script.args_to]
                # 在处理批次之前运行脚本
                script.before_process_batch(p, *script_args, **kwargs)
            except Exception:
                # 报告错误信息
                errors.report(f"Error running before_process_batch: {script.filename}", exc_info=True)

    # 在额外网络激活后运行脚本
    def after_extra_networks_activate(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.after_extra_networks_activate(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running after_extra_networks_activate: {script.filename}", exc_info=True)

    # 处理批次
    def process_batch(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running process_batch: {script.filename}", exc_info=True)

    # 后处理
    def postprocess(self, p, processed):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess(p, processed, *script_args)
            except Exception:
                errors.report(f"Error running postprocess: {script.filename}", exc_info=True)

    # 批次后处理
    def postprocess_batch(self, p, images, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_batch(p, *script_args, images=images, **kwargs)
            except Exception:
                errors.report(f"Error running postprocess_batch: {script.filename}", exc_info=True)
    # 对批处理列表进行后处理，遍历所有的alwayson_scripts
    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, **kwargs):
        for script in self.alwayson_scripts:
            try:
                # 获取当前脚本所需的参数
                script_args = p.script_args[script.args_from:script.args_to]
                # 调用脚本的postprocess_batch_list方法进行后处理
                script.postprocess_batch_list(p, pp, *script_args, **kwargs)
            except Exception:
                # 报告后处理过程中的错误
                errors.report(f"Error running postprocess_batch_list: {script.filename}", exc_info=True)

    # 对图像进行后处理，遍历所有的alwayson_scripts
    def postprocess_image(self, p, pp: PostprocessImageArgs):
        for script in self.alwayson_scripts:
            try:
                # 获取当前脚本所需的参数
                script_args = p.script_args[script.args_from:script.args_to]
                # 调用脚本的postprocess_image方法进行后处理
                script.postprocess_image(p, pp, *script_args)
            except Exception:
                # 报告后处理过程中的错误
                errors.report(f"Error running postprocess_image: {script.filename}", exc_info=True)

    # 在组件之前运行，根据elem_id获取对应的回调函数和脚本
    def before_component(self, component, **kwargs):
        for callback, script in self.on_before_component_elem_id.get(kwargs.get("elem_id"), []):
            try:
                # 调用回调函数，传入OnComponent对象
                callback(OnComponent(component=component))
            except Exception:
                # 报告运行on_before_component时的错误
                errors.report(f"Error running on_before_component: {script.filename}", exc_info=True)

        # 遍历所有的脚本
        for script in self.scripts:
            try:
                # 调用脚本的before_component方法
                script.before_component(component, **kwargs)
            except Exception:
                # 报告运行before_component时的错误
                errors.report(f"Error running before_component: {script.filename}", exc_info=True)
    # 在组件之后执行回调函数和脚本
    def after_component(self, component, **kwargs):
        # 遍历与组件元素ID匹配的回调函数和脚本
        for callback, script in self.on_after_component_elem_id.get(component.elem_id, []):
            try:
                # 执行回调函数，传入组件作为参数
                callback(OnComponent(component=component))
            except Exception:
                # 报告执行回调函数时出现的错误
                errors.report(f"Error running on_after_component: {script.filename}", exc_info=True)

        # 遍历所有脚本对象
        for script in self.scripts:
            try:
                # 在组件之后执行脚本
                script.after_component(component, **kwargs)
            except Exception:
                # 报告执行脚本时出现的错误
                errors.report(f"Error running after_component: {script.filename}", exc_info=True)

    # 根据标题获取脚本对象
    def script(self, title):
        return self.title_map.get(title.lower())

    # 重新加载脚本源文件
    def reload_sources(self, cache):
        # 遍历所有脚本对象
        for si, script in list(enumerate(self.scripts)):
            args_from = script.args_from
            args_to = script.args_to
            filename = script.filename

            # 从缓存中获取模块，如果不存在则加载模块
            module = cache.get(filename, None)
            if module is None:
                module = script_loading.load_module(script.filename)
                cache[filename] = module

            # 遍历模块中的类，如果是 Script 的子类则替换原有脚本对象
            for script_class in module.__dict__.values():
                if type(script_class) == type and issubclass(script_class, Script):
                    self.scripts[si] = script_class()
                    self.scripts[si].filename = filename
                    self.scripts[si].args_from = args_from
                    self.scripts[si].args_to = args_to

    # 在 hr 标签之前执行脚本
    def before_hr(self, p):
        # 遍历所有始终执行的脚本对象
        for script in self.alwayson_scripts:
            try:
                # 获取脚本参数并执行脚本
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_hr(p, *script_args)
            except Exception:
                # 报告执行脚本时出现的错误
                errors.report(f"Error running before_hr: {script.filename}", exc_info=True)
    # 设置脚本，根据参数 p 和 is_ui 来执行不同的操作
    def setup_scrips(self, p, *, is_ui=True):
        # 遍历 self.alwayson_scripts 列表中的每个脚本
        for script in self.alwayson_scripts:
            # 如果不是 UI 操作，并且脚本只设置为 UI 操作，则跳过当前循环
            if not is_ui and script.setup_for_ui_only:
                continue

            # 尝试执行脚本的设置操作
            try:
                # 从 p.script_args 中获取脚本参数的子集
                script_args = p.script_args[script.args_from:script.args_to]
                # 调用脚本的 setup 方法，传入参数 p 和脚本参数
                script.setup(p, *script_args)
            # 捕获任何异常
            except Exception:
                # 报告错误信息，包括出错的脚本文件名
                errors.report(f"Error running setup: {script.filename}", exc_info=True)
# 定义全局变量，用于存储不同类型的脚本运行器对象
scripts_txt2img: ScriptRunner = None
scripts_img2img: ScriptRunner = None
scripts_postproc: scripts_postprocessing.ScriptPostprocessingRunner = None
scripts_current: ScriptRunner = None

# 定义函数，重新加载脚本的主体部分
def reload_script_body_only():
    # 创建一个空字典用于缓存
    cache = {}
    # 重新加载文本到图像脚本的源文件
    scripts_txt2img.reload_sources(cache)
    # 重新加载图像到图像脚本的源文件
    scripts_img2img.reload_sources(cache)

# 将 load_scripts 函数赋值给 reload_scripts 变量，用于向后兼容
reload_scripts = load_scripts  # compatibility alias
```