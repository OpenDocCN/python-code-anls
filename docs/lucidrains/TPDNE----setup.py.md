# `.\lucidrains\TPDNE\setup.py`

```py
# 导入设置工具和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  name = 'TPDNE-utils',  # 包的名称
  packages = find_packages(exclude=[]),  # 查找所有包
  version = '0.0.11',  # 版本号
  license='MIT',  # 许可证
  description = 'TPDNE',  # 描述
  include_package_data = True,  # 包含所有数据文件
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/TPDNE',  # 项目链接
  keywords = [
    'thispersondoesnotexist'  # 关键词
  ],
  install_requires = [  # 安装依赖
    'beartype',
    'einops>=0.6',
    'jinja2',
    'numpy',
    'pillow'
  ],
  classifiers=[  # 分类
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```