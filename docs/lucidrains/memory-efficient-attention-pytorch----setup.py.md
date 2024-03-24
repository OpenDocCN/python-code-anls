# `.\lucidrains\memory-efficient-attention-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'memory-efficient-attention-pytorch',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.1.6',  # 版本号
  license='MIT',  # 许可证
  description = 'Memory Efficient Attention - Pytorch',  # 描述
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/memory-efficient-attention-pytorch',  # 项目链接
  keywords = [
    'artificial intelligence',  # 关键词
    'deep learning',  # 关键词
    'attention-mechanism'  # 关键词
  ],
  install_requires=[
    'einops>=0.4.1',  # 安装所需的依赖项
    'torch>=1.6'    # 安装所需的依赖项
  ],
  setup_requires=[
    'pytest-runner',  # 安装设置所需的依赖项
  ],
  tests_require=[
    'pytest'  # 安装测试所需的依赖项
  ],
  classifiers=[
    'Development Status :: 4 - Beta',  # 分类器
    'Intended Audience :: Developers',  # 分类器
    'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 分类器
    'License :: OSI Approved :: MIT License',  # 分类器
    'Programming Language :: Python :: 3.8',  # 分类器
  ],
)
```