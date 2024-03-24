# `.\lucidrains\robotic-transformer-pytorch\setup.py`

```
# 导入设置工具和查找包函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'robotic-transformer-pytorch', # 包名
  packages = find_packages(exclude=[]), # 查找所有包
  version = '0.2.1', # 版本号
  license='MIT', # 许可证
  description = 'Robotic Transformer - Pytorch', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  long_description_content_type = 'text/markdown', # 长描述内容类型
  url = 'https://github.com/lucidrains/robotic-transformer-pytorch', # 项目链接
  keywords = [ # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'robotics'
  ],
  install_requires=[ # 安装依赖
    'classifier-free-guidance-pytorch>=0.4.0',
    'einops>=0.7',
    'torch>=2.0',
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