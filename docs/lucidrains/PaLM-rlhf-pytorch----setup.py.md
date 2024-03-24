# `.\lucidrains\PaLM-rlhf-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'PaLM-rlhf-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.2.1',  # 版本号
  license='MIT',  # 许可证
  description = 'PaLM + Reinforcement Learning with Human Feedback - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/PaLM-rlhf-pytorch',  # URL
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'reinforcement learning',
    'human feedback'
  ],
  install_requires=[  # 安装依赖
    'accelerate',
    'beartype',
    'einops>=0.6',
    'lion-pytorch',
    'torch>=1.6',
    'tqdm'
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