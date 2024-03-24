# `.\lucidrains\soft-moe-pytorch\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置软件包的元数据
setup(
  name = 'soft-moe-pytorch', # 软件包的名称
  packages = find_packages(exclude=[]), # 查找所有包
  version = '0.1.7', # 版本号
  license='MIT', # 许可证
  description = 'Soft MoE - Pytorch', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/soft-moe-pytorch', # 项目链接
  keywords = [
    'artificial intelligence', # 关键词：人工智能
    'deep learning', # 关键词：深度学习
    'mixture of experts' # 关键词：专家混合
  ],
  install_requires=[
    'einops>=0.6.1', # 安装所需的依赖项：einops 版本大于等于 0.6.1
    'torch>=2.0' # 安装所需的依赖项：torch 版本大于等于 2.0
  ],
  classifiers=[
    'Development Status :: 4 - Beta', # 分类器：开发状态为 Beta
    'Intended Audience :: Developers', # 分类器：面向的受众为开发者
    'Topic :: Scientific/Engineering :: Artificial Intelligence', # 分类器：主题为科学/工程 - 人工智能
    'License :: OSI Approved :: MIT License', # 分类器：许可证为 MIT 许可证
    'Programming Language :: Python :: 3.6', # 分类器：编程语言为 Python 3.6
  ],
)
```