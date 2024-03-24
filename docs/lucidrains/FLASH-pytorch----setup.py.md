# `.\lucidrains\FLASH-pytorch\setup.py`

```py
# 导入设置安装和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'FLASH-pytorch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.1.9',
  # 许可证类型
  license='MIT',
  # 包的描述
  description = 'FLASH - Transformer Quality in Linear Time - Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目的URL
  url = 'https://github.com/lucidrains/FLASH-pytorch',
  # 关键词列表
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.4',
    'rotary-embedding-torch>=0.1.5',
    'torch>=1.9',
  ],
  # 分类标签
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```