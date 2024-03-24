# `.\lucidrains\musiclm-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'musiclm-pytorch', # 包的名称
  packages = find_packages(exclude=[]), # 查找所有包
  version = '0.2.8', # 版本号
  license='MIT', # 许可证
  description = 'MusicLM - AudioLM + Audio CLIP to text to music synthesis', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/musiclm-pytorch', # 项目链接
  keywords = [ # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'text to music',
    'contrastive learning'
  ],
  install_requires=[ # 安装依赖列表
    'accelerate',
    'audiolm-pytorch>=0.17.0',
    'beartype',
    'einops>=0.6',
    'lion-pytorch',
    'vector-quantize-pytorch>=1.0.0',
    'x-clip',
    'torch>=1.12',
    'torchaudio'
  ],
  classifiers=[ # 分类器列表
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```