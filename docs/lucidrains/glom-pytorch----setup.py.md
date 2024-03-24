# `.\lucidrains\glom-pytorch\setup.py`

```
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'glom-pytorch', # 包的名称
  packages = find_packages(), # 查找所有包
  version = '0.0.14', # 版本号
  license='MIT', # 许可证
  description = 'Glom - Pytorch', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  url = 'https://github.com/lucidrains/glom-pytorch', # 项目链接
  keywords = [
    'artificial intelligence', # 关键词
    'deep learning'
  ],
  install_requires=[
    'einops>=0.3', # 安装所需的依赖包
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta', # 分类器
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```