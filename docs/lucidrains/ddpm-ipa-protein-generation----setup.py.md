# `.\lucidrains\ddpm-ipa-protein-generation\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'ddpm-ipa-protein-generation',  # 包名
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.0.1',  # 版本号
  license='MIT',  # 许可证
  description = 'DDPM + Invariant Point Attention - Protein Generation',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/ddpm-ipa-protein-generation',  # URL
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'attention mechanism',
    'geometric deep learning',
    'denoising diffusion probabilistic models'
  ],
  install_requires=[  # 安装依赖
    'invariant-point-attention>=0.2.1',
    'einops>=0.4',
    'torch>=1.6',
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