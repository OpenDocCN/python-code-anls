# `.\lucidrains\n-grammer-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'n-grammer-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.0.14',  # 版本号
  license='MIT',  # 许可证
  description = 'N-Grammer - Pytorch',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/n-grammer-pytorch',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'n-grams',
    'memory'
  ],
  install_requires=[  # 安装依赖
    'einops>=0.3',
    'sympy',
    'torch>=1.6'
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