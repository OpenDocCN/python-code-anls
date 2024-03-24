# `.\lucidrains\unet-stylegan2\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'unet_stylegan2',  # 包的名称
  packages = find_packages(),  # 查找并包含所有包
  scripts=['bin/unet_stylegan2'],  # 包含可执行脚本
  version = '0.5.1',  # 版本号
  license='GPLv3+',  # 许可证
  description = 'StyleGan2 with UNet Discriminator, in Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/unet-stylegan2',  # 项目链接
  keywords = ['generative adversarial networks', 'artificial intelligence'],  # 关键词
  install_requires=[  # 安装依赖
      'fire',
      'numpy',
      'retry',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
      'linear_attention_transformer>=0.12.1'
  ],
  classifiers=[  # 分类
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)
```