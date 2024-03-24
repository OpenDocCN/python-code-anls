# `.\lucidrains\st-moe-pytorch\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  name = 'st-moe-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.1.7',  # 版本号
  license='MIT',  # 许可证
  description = 'ST - Mixture of Experts - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/st-moe-pytorch',  # URL
  keywords = [
    'artificial intelligence',  # 关键词
    'deep learning',  # 关键词
    'mixture of experts'  # 关键词
  ],
  install_requires=[
    'beartype',  # 安装所需的包
    'CoLT5-attention>=0.10.15',  # 安装所需的包
    'einops>=0.6',  # 安装所需的包
    'torch>=2.0',  # 安装所需的包
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 分类器
    'Intended Audience :: Developers',  # 分类器
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 分类器
    'License :: OSI Approved :: MIT License',  # 分类器
    'Programming Language :: Python :: 3.6',  # 分类器
  ],
)
```