# `.\lucidrains\nuwa-pytorch\setup.py`

```py
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'nuwa-pytorch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 包含所有数据文件
  include_package_data = True,
  # 版本号
  version = '0.7.8',
  # 许可证类型
  license='MIT',
  # 包的描述
  description = 'NÜWA - Pytorch',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/nuwa-pytorch',
  # 关键词列表
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  # 安装依赖包
  install_requires=[
    'einops>=0.4.1',
    'ftfy',
    'pillow',
    'regex',
    'torch>=1.6',
    'torchvision',
    'tqdm',
    'unfoldNd',
    'vector-quantize-pytorch>=0.4.10'
  ],
  # 分类标签列表
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```