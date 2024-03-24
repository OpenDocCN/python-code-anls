# `.\lucidrains\ddpm-proteins\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'ddpm-proteins',  # 包名
  packages = find_packages(),  # 查找所有包
  version = '0.0.11',  # 版本号
  license='MIT',  # 许可证
  description = 'Denoising Diffusion Probabilistic Models - for Proteins - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/ddpm-proteins',  # 项目链接
  keywords = [  # 关键词
    'artificial intelligence',
    'generative models',
    'proteins'
  ],
  install_requires=[  # 依赖的包
    'einops',
    'matplotlib',
    'numpy',
    'pillow',
    'proDy',
    'scipy',
    'sidechainnet',
    'seaborn',
    'torch',
    'torchvision',
    'tqdm',
    'wandb'
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