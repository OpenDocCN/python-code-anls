# `.\lucidrains\adjacent-attention-network\setup.py`

```py
# 导入设置安装和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'adjacent-attention-pytorch', # 包的名称
  packages = find_packages(), # 查找所有包
  version = '0.0.12', # 版本号
  license='MIT', # 许可证
  description = 'Adjacent Attention Network - Pytorch', # 描述
  long_description_content_type = 'text/markdown', # 长描述内容类型
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  url = 'https://github.com/lucidrains/adjacent-attention-pytorch', # 项目链接
  keywords = [ # 关键词列表
    'artificial intelligence',
    'attention mechanism',
    'graph neural network',
    'transformers'
  ],
  install_requires=[ # 安装依赖
    'einops>=0.3',
    'torch>=1.6',
    'isab-pytorch<0.2'
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