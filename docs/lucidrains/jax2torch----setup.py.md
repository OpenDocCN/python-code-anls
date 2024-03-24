# `.\lucidrains\jax2torch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  # 包的名称
  name = 'jax2torch',
  # 查找所有包，不排除任何包
  packages = find_packages(exclude=[]),
  # 版本号
  version = '0.0.7',
  # 许可证类型
  license='MIT',
  # 描述信息
  description = 'Jax 2 Torch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/jax2torch',
  # 关键词列表
  keywords = [
    'jax',
    'pytorch'
  ],
  # 安装依赖项
  install_requires=[
    'torch>=1.6',
    'jax>=0.2.20'
  ],
  # 分类器列表
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```