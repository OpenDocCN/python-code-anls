# `.\lucidrains\voicebox-pytorch\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'voicebox-pytorch',  # 包名
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.5.0',  # 版本号
  license='MIT',  # 许可证
  description = 'Voicebox - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/voicebox-pytorch',  # URL
  keywords = [  # 关键词
    'artificial intelligence',
    'deep learning',
    'text to speech'
  ],
  install_requires=[  # 安装依赖
    'accelerate',
    'audiolm-pytorch>=1.2.28',
    'naturalspeech2-pytorch>=0.1.8',
    'beartype',
    'einops>=0.6.1',
    'gateloop-transformer>=0.2.4',
    'spear-tts-pytorch>=0.4.0',
    'torch>=2.0',
    'torchdiffeq',
    'torchode',
    'vocos'
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