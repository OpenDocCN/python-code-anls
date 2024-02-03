# `stable-diffusion-webui\modules\script_loading.py`

```
# 导入必要的模块
import os
import importlib.util

# 从自定义模块中导入错误处理模块
from modules import errors

# 加载指定路径的模块并返回
def load_module(path):
    # 根据路径创建模块规范
    module_spec = importlib.util.spec_from_file_location(os.path.basename(path), path)
    # 根据模块规范创建模块对象
    module = importlib.util.module_from_spec(module_spec)
    # 执行模块对象的代码
    module_spec.loader.exec_module(module)

    return module

# 预加载扩展模块
def preload_extensions(extensions_dir, parser, extension_list=None):
    # 如果指定的扩展目录不存在，则直接返回
    if not os.path.isdir(extensions_dir):
        return

    # 如果未指定扩展列表，则获取目录下所有文件名作为扩展列表
    extensions = extension_list if extension_list is not None else os.listdir(extensions_dir)
    # 遍历扩展列表
    for dirname in sorted(extensions):
        # 构建预加载脚本的路径
        preload_script = os.path.join(extensions_dir, dirname, "preload.py")
        # 如果预加载脚本不存在，则跳过当前循环
        if not os.path.isfile(preload_script):
            continue

        try:
            # 加载预加载脚本作为模块
            module = load_module(preload_script)
            # 如果模块中存在名为 'preload' 的属性，则调用该属性并传入解析器对象
            if hasattr(module, 'preload'):
                module.preload(parser)

        # 捕获任何异常并报告错误信息
        except Exception:
            errors.report(f"Error running preload() for {preload_script}", exc_info=True)
```