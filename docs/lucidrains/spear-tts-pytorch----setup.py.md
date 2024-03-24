# `.\lucidrains\spear-tts-pytorch\setup.py`

```py
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包名
  name = 'spear-tts-pytorch',
  # 查找包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.4.8',
  # 许可证
  license='MIT',
  # 描述
  description = 'Spear-TTS - Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 项目链接
  url = 'https://github.com/lucidrains/spear-tts-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'text-to-speech'
  ],
  # 安装依赖
  install_requires=[
    'audiolm-pytorch>=1.2.8',
    'beartype',
    'einops>=0.6.1',
    'rotary-embedding-torch>=0.3.0',
    'torch>=1.6',
    'tqdm',
    'x-clip>=0.12.2'
  ],
  # 分类
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```