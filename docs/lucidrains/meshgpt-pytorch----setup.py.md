# `.\lucidrains\meshgpt-pytorch\setup.py`

```py
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('meshgpt_pytorch/version.py').read())

# 设置包的元信息
setup(
  name = 'meshgpt-pytorch',  # 包名
  packages = find_packages(exclude=[]),  # 查找包
  version = __version__,  # 版本号
  license='MIT',  # 许可证
  description = 'MeshGPT Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/meshgpt-pytorch',  # URL
  keywords = [  # 关键词
    'artificial intelligence',
    'deep learning',
    'attention mechanisms',
    'transformers',
    'mesh generation'
  ],
  install_requires=[  # 安装依赖
    'accelerate>=0.25.0',
    'beartype',
    'classifier-free-guidance-pytorch>=0.5.1',
    'einops>=0.7.0',
    'einx[torch]>=0.1.3',
    'ema-pytorch',
    'local-attention>=1.9.0',
    'gateloop-transformer>=0.2.2',
    'numpy',
    'pytorch-custom-utils>=0.0.9',
    'taylor-series-linear-attention>=0.1.6',
    'torch>=2.1',
    'torch_geometric',
    'torchtyping',
    'tqdm',
    'vector-quantize-pytorch>=1.12.8',
    'x-transformers>=1.26.0',
  ],
  classifiers=[  # 分类器
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```