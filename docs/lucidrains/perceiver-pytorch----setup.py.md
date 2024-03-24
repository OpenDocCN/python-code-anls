# `.\lucidrains\perceiver-pytorch\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'perceiver-pytorch',  # 包名
  packages = find_packages(),  # 查找所有包
  version = '0.8.8',  # 版本号
  license='MIT',  # 许可证
  description = 'Perceiver - Pytorch',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/perceiver-pytorch',  # 项目链接
  keywords = [  # 关键词列表
    'artificial intelligence',
    'deep learning',
    'transformer',
    'attention mechanism'
  ],
  install_requires=[  # 安装依赖
    'einops>=0.3',
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