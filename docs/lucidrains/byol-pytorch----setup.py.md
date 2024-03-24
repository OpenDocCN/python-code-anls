# `.\lucidrains\byol-pytorch\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'byol-pytorch',
  # 查找并包含除了'examples'之外的所有包
  packages = find_packages(exclude=['examples']),
  # 版本号
  version = '0.8.0',
  # 许可证类型
  license='MIT',
  # 描述信息
  description = 'Self-supervised contrastive learning made simple',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/byol-pytorch',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 关键词列表
  keywords = [
      'self-supervised learning',
      'artificial intelligence'
  ],
  # 安装依赖
  install_requires=[
      'accelerate',
      'beartype',
      'torch>=1.6',
      'torchvision>=0.8'
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