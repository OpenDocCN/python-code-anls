# `.\lucidrains\DALLE-pytorch\setup.py`

```py
# 导入设置工具和查找包工具
from setuptools import setup, find_packages
# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('dalle_pytorch/version.py').read())

# 设置包的元信息
setup(
  # 包名
  name = 'dalle-pytorch',
  # 查找所有包
  packages = find_packages(),
  # 包含所有数据文件
  include_package_data = True,
  # 版本号
  version = __version__,
  # 许可证
  license='MIT',
  # 描述
  description = 'DALL-E - Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目链接
  url = 'https://github.com/lucidrains/dalle-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'text-to-image'
  ],
  # 安装依赖
  install_requires=[
    'axial_positional_embedding',
    'DALL-E',
    'einops>=0.3.2',
    'ftfy',
    'packaging',
    'pillow',
    'regex',
    'rotary-embedding-torch',
    'taming-transformers-rom1504',
    'tokenizers',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'tqdm',
    'youtokentome',
    'WebDataset'
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