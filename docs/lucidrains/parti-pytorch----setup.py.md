# `.\lucidrains\parti-pytorch\setup.py`

```
# 导入设置工具和查找包
from setuptools import setup, find_packages
# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('parti_pytorch/version.py').read())

# 设置包的元信息
setup(
  name = 'parti-pytorch',  # 包名
  packages = find_packages(exclude=[]),  # 查找包
  version = __version__,  # 版本号
  license='MIT',  # 许可证
  description = 'Parti - Pathways Autoregressive Text-to-Image Model - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/parti-pytorch',  # URL
  keywords = [  # 关键词
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'text-to-image'
  ],
  install_requires=[  # 安装依赖
    'einops>=0.7',
    'einops-exts',
    'ema-pytorch',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'vector-quantize-pytorch>=1.11.8'
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