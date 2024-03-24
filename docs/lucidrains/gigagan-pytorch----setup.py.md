# `.\lucidrains\gigagan-pytorch\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('gigagan_pytorch/version.py').read())

# 设置包的元数据
setup(
  name = 'gigagan-pytorch', # 包名
  packages = find_packages(exclude=[]), # 查找包
  version = __version__, # 版本号
  license='MIT', # 许可证
  description = 'GigaGAN - Pytorch', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/ETSformer-pytorch', # URL
  keywords = [ # 关键词
    'artificial intelligence',
    'deep learning',
    'generative adversarial networks'
  ],
  install_requires=[ # 安装依赖
    'accelerate',
    'beartype',
    'einops>=0.6',
    'ema-pytorch',
    'kornia',
    'numerize',
    'open-clip-torch>=2.0.0,<3.0.0',
    'pillow',
    'torch>=1.6',
    'torchvision',
    'tqdm'
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