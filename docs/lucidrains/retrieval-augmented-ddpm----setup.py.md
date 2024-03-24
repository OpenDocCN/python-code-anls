# `.\lucidrains\retrieval-augmented-ddpm\setup.py`

```py
# 导入设置安装和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'retrieval-augmented-ddpm',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.0.1',  # 版本号
  license='MIT',  # 许可证
  description = 'Retrieval-Augmented Denoising Diffusion Probabilistic Models',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/retrieval-augmented-ddpm',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',    
    'denoising diffusion',
    'retrieval'
  ],
  install_requires=[  # 安装依赖的包
    'clip-retrieval',
    'einops>=0.4',
    'torch>=1.6',
  ],
  classifiers=[  # 分类器列表
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```