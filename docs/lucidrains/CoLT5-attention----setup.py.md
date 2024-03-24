# `.\lucidrains\CoLT5-attention\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'CoLT5-attention',
  # 查找并包含所有包
  packages = find_packages(),
  # 版本号
  version = '0.10.20',
  # 许可证类型
  license='MIT',
  # 描述
  description = 'Conditionally Routed Attention',
  # 长描述内容类型
  long_description_content_type = 'text/markdown',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/CoLT5-attention',
  # 关键词
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'dynamic routing'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.6.1',
    'local-attention>=1.8.6',
    'packaging',
    'torch>=1.10'
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