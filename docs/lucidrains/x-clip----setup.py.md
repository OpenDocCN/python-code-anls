# `.\lucidrains\x-clip\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包的名称
  name = 'x-clip',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 包含所有数据文件
  include_package_data = True,
  # 版本号
  version = '0.14.4',
  # 许可证类型
  license='MIT',
  # 描述
  description = 'X-CLIP',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/x-clip',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 关键词列表
  keywords = [
    'artificial intelligence',
    'deep learning',
    'contrastive learning',
    'CLIP',
  ],
  # 安装依赖
  install_requires=[
    'beartype',
    'einops>=0.6',
    'ftfy',
    'regex',
    'torch>=1.6',
    'torchvision'
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