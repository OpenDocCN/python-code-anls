# `.\lucidrains\nystrom-attention\setup.py`

```
# 导入设置和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'nystrom-attention',
  # 查找所有包
  packages = find_packages(),
  # 版本号
  version = '0.0.12',
  # 许可证
  license='MIT',
  # 描述
  description = 'Nystrom Attention - Pytorch',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/nystrom-attention',
  # 关键词
  keywords = [
    'artificial intelligence',
    'attention mechanism'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.7.0',
    'torch>=2.0'
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