# `.\lucidrains\bidirectional-cross-attention\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'bidirectional-cross-attention',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.0.5',  # 版本号
  license='MIT',  # 许可证
  description = 'Bidirectional Cross Attention',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/bidirectional-cross-attention',  # 项目链接
  keywords = [
    'artificial intelligence',  # 关键词
    'deep learning',  # 关键词
    'attention mechanism'  # 关键词
  ],
  install_requires=[
    'einops>=0.7',  # 安装所需的依赖包
    'torch>=2.0',  # 安装所需的依赖包
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 分类器
    'Intended Audience :: Developers',  # 分类器
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 分类器
    'License :: OSI Approved :: MIT License',  # 分类器
    'Programming Language :: Python :: 3.6',  # 分类器
  ],
)
```