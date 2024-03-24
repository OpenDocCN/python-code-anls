# `.\lucidrains\vit-pytorch\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'vit-pytorch',
  # 查找除了 'examples' 文件夹之外的所有包
  packages = find_packages(exclude=['examples']),
  # 版本号
  version = '1.6.5',
  # 许可证类型
  license='MIT',
  # 描述
  description = 'Vision Transformer (ViT) - Pytorch',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/vit-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'image recognition'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.7.0',
    'torch>=1.10',
    'torchvision'
  ],
  # 设置需要的依赖
  setup_requires=[
    'pytest-runner',
  ],
  # 测试需要的依赖
  tests_require=[
    'pytest',
    'torch==1.12.1',
    'torchvision==0.13.1'
  ],
  # 分类
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```