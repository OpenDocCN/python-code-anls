# `stable-diffusion-webui\test\conftest.py`

```
# 导入 os 模块
import os

# 导入 pytest 模块
import pytest

# 导入 base64 模块
import base64

# 获取测试文件路径
test_files_path = os.path.dirname(__file__) + "/test_files"

# 将文件转换为 base64 编码的字符串
def file_to_base64(filename):
    # 以二进制只读方式打开文件
    with open(filename, "rb") as file:
        # 读取文件内容
        data = file.read()

    # 将文件内容进行 base64 编码
    base64_str = str(base64.b64encode(data), "utf-8")
    # 返回以 data:image/png;base64, 开头的 base64 编码字符串
    return "data:image/png;base64," + base64_str

# 定义一个 pytest 的 fixture，作用域为 session，以便不重复读取
def img2img_basic_image_base64() -> str:
    # 返回 img2img_basic.png 文件的 base64 编码字符串
    return file_to_base64(os.path.join(test_files_path, "img2img_basic.png"))

# 定义一个 pytest 的 fixture，作用域为 session，以便不重复读取
def mask_basic_image_base64() -> str:
    # 返回 mask_basic.png 文件的 base64 编码字符串
    return file_to_base64(os.path.join(test_files_path, "mask_basic.png"))
```