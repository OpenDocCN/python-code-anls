# `.\PaddleOCR\ppocr\postprocess\pse_postprocess\pse\__init__.py`

```py
# 版权声明
#
# 版权所有 (c) 2020 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言管理权限和限制。
import sys
import os
import subprocess

# 获取当前 Python 解释器路径
python_path = sys.executable

# 获取当前工作目录
ori_path = os.getcwd()
# 切换到指定目录
os.chdir('ppocr/postprocess/pse_postprocess/pse')
# 调用子进程执行编译命令
if subprocess.call(
        '{} setup.py build_ext --inplace'.format(python_path), shell=True) != 0:
    # 如果编译失败，抛出运行时错误
    raise RuntimeError(
        'Cannot compile pse: {}, if your system is windows, you need to install all the default components of `desktop development using C++` in visual studio 2019+'.
        format(os.path.dirname(os.path.realpath(__file__))))
# 切换回原始工作目录
os.chdir(ori_path)

# 从 pse 模块中导入 pse 函数
from .pse import pse
```