# `.\lucidrains\AoA-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'aoa_pytorch', # 包的名称
  packages = find_packages(exclude=['examples']), # 查找并包含除了 examples 之外的所有包
  version = '0.0.2', # 版本号
  license='MIT', # 许可证信息
  description = 'Attention on Attention - Pytorch', # 包的描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者的邮箱
  url = 'https://github.com/lucidrains/SAoA-pytorch', # 项目的链接
  keywords = [
    'artificial intelligence', # 关键词：人工智能
    'attention mechanism', # 关键词：注意力机制
    'visual question answering' # 关键词：视觉问题回答
  ],
  install_requires=[
    'torch>=1.6', # 安装所需的 torch 版本
    'einops>=0.3' # 安装所需的 einops 版本
  ],
  classifiers=[
    'Development Status :: 4 - Beta', # 开发状态
    'Intended Audience :: Developers', # 预期的受众
    'Topic :: Scientific/Engineering :: Artificial Intelligence', # 主题
    'License :: OSI Approved :: MIT License', # 许可证
    'Programming Language :: Python :: 3.6', # 使用的编程语言版本
  ],
)
```