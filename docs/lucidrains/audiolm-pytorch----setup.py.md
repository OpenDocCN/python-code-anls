# `.\lucidrains\audiolm-pytorch\setup.py`

```py
# 导入设置工具和查找包工具
from setuptools import setup, find_packages
# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('audiolm_pytorch/version.py').read())

# 设置包的元信息
setup(
  name = 'audiolm-pytorch',  # 包名
  packages = find_packages(exclude=[]),  # 查找包
  version = __version__,  # 版本号
  license='MIT',  # 许可证
  description = 'AudioLM - Language Modeling Approach to Audio Generation from Google Research - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/audiolm-pytorch',  # URL
  keywords = [  # 关键词
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'audio generation'
  ],
  install_requires=[  # 安装依赖
    'accelerate>=0.24.0',
    'beartype>=0.16.1',
    'einops>=0.7.0',
    'ema-pytorch>=0.2.2',
    'encodec',
    'fairseq',
    'wandb',
    'gateloop-transformer>=0.2.3',
    'joblib',
    'local-attention>=1.9.0',
    'pytorch-warmup',
    'scikit-learn',
    'sentencepiece',
    'torch>=2.1',
    'torchaudio',
    'transformers',
    'tqdm',
    'vector-quantize-pytorch>=1.12.5'
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