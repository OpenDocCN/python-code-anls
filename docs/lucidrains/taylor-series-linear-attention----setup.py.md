# `.\lucidrains\taylor-series-linear-attention\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'taylor-series-linear-attention', # 包的名称
  packages = find_packages(exclude=[]), # 查找所有包
  version = '0.1.9', # 版本号
  license='MIT', # 许可证
  description = 'Taylor Series Linear Attention', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/taylor-series-linear-attention', # 项目链接
  keywords = [
    'artificial intelligence', # 关键词
    'deep learning', # 关键词
    'attention mechanism' # 关键词
  ],
  install_requires=[
    'einops>=0.7.0', # 安装所需的依赖包
    'einx', # 安装所需的依赖包
    'rotary-embedding-torch>=0.5.3', # 安装所需的依赖包
    'torch>=2.0', # 安装所需的依赖包
    'torchtyping' # 安装所需的依赖包
  ],
  classifiers=[
    'Development Status :: 4 - Beta', # 分类器
    'Intended Audience :: Developers', # 分类器
    'Topic :: Scientific/Engineering :: Artificial Intelligence', # 分类器
    'License :: OSI Approved :: MIT License', # 分类器
    'Programming Language :: Python :: 3.6', # 分类器
  ],
)
```