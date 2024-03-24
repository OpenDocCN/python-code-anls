# `.\lucidrains\naturalspeech2-pytorch\setup.py`

```py
# 导入设置安装和查找包的函数
from setuptools import setup, find_packages

# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('naturalspeech2_pytorch/version.py').read())

# 设置包的元数据
setup(
  name = 'naturalspeech2-pytorch', # 包名
  packages = find_packages(exclude=[]), # 查找所有包
  version = __version__, # 版本号
  license='MIT', # 许可证
  description = 'Natural Speech 2 - Pytorch', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  include_package_data = True, # 包含包数据
  url = 'https://github.com/lucidrains/naturalspeech2-pytorch', # 项目链接
  keywords = [ # 关键词
    'artificial intelligence',
    'deep learning',
    'latent diffusion',
    'speech synthesis'
  ],
  install_requires=[ # 安装依赖
    'accelerate',
    'audiolm-pytorch>=0.30.2',
    'beartype',
    'einops>=0.6.1',
    'ema-pytorch',
    'indic-num2words',
    'inflect',
    'local-attention',
    'num2words',
    'pyworld',
    'pydantic<2.0',
    'torch>=1.6',
    'tqdm',
    'vector-quantize-pytorch>=1.4.1'
  ],
  classifiers=[ # 分类器
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```