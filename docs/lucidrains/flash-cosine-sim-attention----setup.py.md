# `.\lucidrains\flash-cosine-sim-attention\setup.py`

```py
# 导入必要的库
import sys
from functools import lru_cache
from subprocess import DEVNULL, call
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 以下代码来源于指定链接，用于获取版本号
exec(open('flash_cosine_sim_attention/version.py').read())

# 检查是否存在 CUDA 工具包
@lru_cache(None)
def cuda_toolkit_available():
  try:
    # 尝试调用 nvcc 命令，如果成功则返回 True
    call(["nvcc"], stdout = DEVNULL, stderr = DEVNULL)
    return True
  except FileNotFoundError:
    # 如果未找到 nvcc 命令，则返回 False
    return False

# 编译参数
def compile_args():
  args = ["-fopenmp", "-ffast-math"]
  if sys.platform == "darwin":
    # 如果是 macOS 系统，则添加额外的编译参数
    args = ["-Xpreprocessor", *args]
  return args

# 扩展模块
def ext_modules():
  if not cuda_toolkit_available():
    # 如果 CUDA 工具包不可用，则返回空列表
    return []

  return [
    CUDAExtension(
      __cuda_pkg_name__,
      sources = ["flash_cosine_sim_attention/flash_cosine_sim_attention_cuda.cu"]
    )
  ]

# 主要设置代码
setup(
  name = 'flash-cosine-sim-attention',
  packages = find_packages(exclude=[]),
  version = __version__,
  license='MIT',
  description = 'Flash Cosine Similarity Attention',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/flash-cosine-sim-attention',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanism'
  ],
  install_requires=[
    'torch>=1.10'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  ext_modules = ext_modules(),
  cmdclass = {"build_ext": BuildExtension},
  include_package_data = True,
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```