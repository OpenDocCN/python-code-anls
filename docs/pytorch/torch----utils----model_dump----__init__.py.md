# `.\pytorch\torch\utils\model_dump\__init__.py`

```
#!/usr/bin/env python3
# mypy: allow-untyped-defs
"""
model_dump: a one-stop shop for TorchScript model inspection.

The goal of this tool is to provide a simple way to extract lots of
useful information from a TorchScript model and make it easy for humans
to consume.  It (mostly) replaces zipinfo, common uses of show_pickle,
and various ad-hoc analysis notebooks.

The tool extracts information from the model and serializes it as JSON.
That JSON can then be rendered by an HTML+JS page, either by
loading the JSON over HTTP or producing a fully self-contained page
with all of the code and data burned-in.
"""

# Maintainer notes follow.
"""
The implementation strategy has tension between 3 goals:
- Small file size.
- Fully self-contained.
- Easy, modern JS environment.
Using Preact and HTM achieves 1 and 2 with a decent result for 3.
However, the models I tested with result in ~1MB JSON output,
so even using something heavier like full React might be tolerable
if the build process can be worked out.

One principle I have followed that I think is very beneficial
is to keep the JSON data as close as possible to the model
and do most of the rendering logic on the client.
This makes for easier development (just refresh, usually),
allows for more laziness and dynamism, and lets us add more
views of the same data without bloating the HTML file.

Currently, this code doesn't actually load the model or even
depend on any part of PyTorch.  I don't know if that's an important
feature to maintain, but it's probably worth preserving the ability
to run at least basic analysis on models that cannot be loaded.

I think the easiest way to develop this code is to cd into model_dump and
run "python -m http.server", then load http://localhost:8000/skeleton.html
in the browser.  In another terminal, run
"python -m torch.utils.model_dump --style=json FILE > \
    torch/utils/model_dump/model_info.json"
every time you update the Python code or model.
When you update JS, just refresh.

Possible improvements:
    - Fix various TODO comments in this file and the JS.
    - Make the HTML much less janky, especially the auxiliary data panel.
    - Make the auxiliary data panel start small, expand when
      data is available, and have a button to clear/contract.
    - Clean up the JS.  There's a lot of copypasta because
      I don't really know how to use Preact.
    - Make the HTML render and work nicely inside a Jupyter notebook.
    - Add the ability for JS to choose the URL to load the JSON based
      on the page URL (query or hash).  That way we could publish the
      inlined skeleton once and have it load various JSON blobs.
    - Add a button to expand all expandable sections so ctrl-F works well.
    - Add hyperlinking from data to code, and code to code.
    - Add hyperlinking from debug info to Diffusion.
    - Make small tensor contents available.
    - Do something nice for quantized models
      (they probably don't work at all right now).
"""

import argparse
import io  # 导入io模块，提供基本的输入输出操作
import json  # 导入json模块，用于处理JSON格式数据
import os  # 导入os模块，提供与操作系统交互的功能
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import pprint  # 导入pprint模块，用于美观打印数据结构
import re  # 导入re模块，提供正则表达式匹配功能
import sys  # 导入sys模块，提供对解释器相关的操作
import urllib.parse  # 导入urllib.parse模块，用于URL解析
import zipfile  # 导入zipfile模块，用于ZIP文件操作
from pathlib import Path  # 从pathlib模块导入Path类，用于处理文件路径
from typing import Dict  # 导入Dict类型提示，用于类型注解

import torch.utils.show_pickle  # 导入torch.utils.show_pickle模块中的内容

DEFAULT_EXTRA_FILE_SIZE_LIMIT = 16 * 1024  # 设置默认额外文件大小限制为16KB

__all__ = ['get_storage_info', 'hierarchical_pickle', 'get_model_info', 'get_inline_skeleton',
           'burn_in_info', 'get_info_and_burn_skeleton']

def get_storage_info(storage):
    # 确保storage是torch.utils.show_pickle.FakeObject的实例
    assert isinstance(storage, torch.utils.show_pickle.FakeObject)
    # 确保storage的module属性为"pers"
    assert storage.module == "pers"
    # 确保storage的name属性为"obj"
    assert storage.name == "obj"
    # 确保storage的state属性为None
    assert storage.state is None
    # 确保storage的args属性是一个元组，并且长度为1
    assert isinstance(storage.args, tuple)
    assert len(storage.args) == 1
    sa = storage.args[0]
    # 确保sa是一个元组，并且长度为5
    assert isinstance(sa, tuple)
    assert len(sa) == 5
    # 确保sa的第一个元素为"storage"
    assert sa[0] == "storage"
    # 确保sa的第二个元素是torch.utils.show_pickle.FakeClass的实例
    assert isinstance(sa[1], torch.utils.show_pickle.FakeClass)
    # 确保sa的第二个元素的module属性为"torch"，并且name属性以"Storage"结尾
    assert sa[1].module == "torch"
    assert sa[1].name.endswith("Storage")
    # 构建存储信息列表，包括存储类别和其余参数
    storage_info = [sa[1].name.replace("Storage", "")] + list(sa[2:])
    return storage_info

def hierarchical_pickle(data):
    # 如果data是基本类型(bool, int, float, str, None)，直接返回
    if isinstance(data, (bool, int, float, str, type(None))):
        return data
    # 如果data是列表，递归处理列表中的每个元素
    if isinstance(data, list):
        return [hierarchical_pickle(d) for d in data]
    # 如果data是元组，转换为字典形式标记为元组
    if isinstance(data, tuple):
        return {
            "__tuple_values__": hierarchical_pickle(list(data)),
        }
    # 如果data是字典，转换为字典形式标记为字典，并提取其键和值
    if isinstance(data, dict):
        return {
            "__is_dict__": True,
            "keys": hierarchical_pickle(list(data.keys())),
            "values": hierarchical_pickle(list(data.values())),
        }
    # 如果data类型无法处理，抛出异常
    raise Exception(f"Can't prepare data of type for JS: {type(data)}")  # noqa: TRY002

def get_model_info(
        path_or_file,
        title=None,
        extra_file_size_limit=DEFAULT_EXTRA_FILE_SIZE_LIMIT):
    """Get JSON-friendly information about a model.

    The result is suitable for being saved as model_info.json,
    or passed to burn_in_info.
    """

    # 根据path_or_file的类型设置默认标题和文件大小
    if isinstance(path_or_file, os.PathLike):
        default_title = os.fspath(path_or_file)
        file_size = path_or_file.stat().st_size  # type: ignore[attr-defined]
    elif isinstance(path_or_file, str):
        default_title = path_or_file
        file_size = Path(path_or_file).stat().st_size
    else:
        default_title = "buffer"
        path_or_file.seek(0, io.SEEK_END)
        file_size = path_or_file.tell()
        path_or_file.seek(0)

    # 根据title参数设置标题
    title = title or default_title

    # 返回模型信息的字典形式，包括标题、文件大小和其他信息
    return {"model": dict(
        title=title,
        file_size=file_size,
        version=version,
        zip_files=zip_files,
        interned_strings=list(interned_strings),
        code_files=code_files,
        model_data=model_data,
        constants=constants,
        extra_files_jsons=extra_files_jsons,
        extra_pickles=extra_pickles,
    )}

def get_inline_skeleton():
    """Get a fully-inlined skeleton of the frontend.

    The returned HTML page has no external network dependencies for code.
    """
    It can load model_info.json over HTTP, or be passed to burn_in_info.
    """

    # 导入 importlib.resources 模块，用于访问包中的资源
    import importlib.resources

    # 读取包中的 skeleton.html 文件内容，存储在 skeleton 变量中
    skeleton = importlib.resources.read_text(__package__, "skeleton.html")

    # 读取包中的 code.js 文件内容，存储在 js_code 变量中
    js_code = importlib.resources.read_text(__package__, "code.js")

    # 遍历 js_module 列表，每次读取对应的模块的二进制数据，并生成 data URL
    for js_module in ["preact", "htm"]:
        js_lib = importlib.resources.read_binary(__package__, f"{js_module}.mjs")
        js_url = "data:application/javascript," + urllib.parse.quote(js_lib)
        # 替换 js_code 中的模块引用 URL，将原始的 https://unpkg.com/{js_module}?module 替换为生成的 js_url
        js_code = js_code.replace(f"https://unpkg.com/{js_module}?module", js_url)

    # 替换 skeleton 中的脚本标签的 src 属性，将其替换为包含 js_code 的脚本内容
    skeleton = skeleton.replace(' src="./code.js">', ">\n" + js_code)

    # 返回替换后的 skeleton，即包含了替换后 JavaScript 代码的 HTML 模板
    return skeleton
# 解析命令行参数，配置参数选项
parser = argparse.ArgumentParser()
parser.add_argument("--style", choices=["json", "html"])  # 支持的输出样式选项为 JSON 或 HTML
parser.add_argument("--title")  # 可选的页面标题参数
parser.add_argument("model")  # 模型文件路径参数
args = parser.parse_args(argv[1:])  # 解析命令行参数，忽略第一个参数（脚本名）

# 获取模型信息，根据命令行提供的模型文件路径和可选的页面标题
info = get_model_info(args.model, title=args.title)

# 根据标准输出或指定的输出流选择输出目标
output = stdout or sys.stdout

# 根据用户指定的样式格式化输出内容
if args.style == "json":
    # 如果选择 JSON 格式，则将模型信息转换为 JSON 格式并写入输出流
    output.write(json.dumps(info, sort_keys=True) + "\n")
elif args.style == "html":
    # 如果选择 HTML 格式，则获取内联骨架并将模型信息嵌入到 HTML 中
    skeleton = get_inline_skeleton()
    page = burn_in_info(skeleton, info)  # 将模型信息烧录到 HTML 骨架中
    output.write(page)  # 将生成的 HTML 页面写入输出流
else:
    # 如果样式参数不是 "json" 或 "html"，则抛出异常
    raise Exception("Invalid style")  # noqa: TRY002
```