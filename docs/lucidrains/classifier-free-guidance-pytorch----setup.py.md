# `.\lucidrains\classifier-free-guidance-pytorch\setup.py`

```py
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元数据
setup(
  name = 'classifier-free-guidance-pytorch',  # 包名
  packages = find_packages(exclude=[]),  # 查找包
  include_package_data = True,  # 包含数据文件
  version = '0.5.3',  # 版本号
  license='MIT',  # 许可证
  description = 'Classifier Free Guidance - Pytorch',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  long_description_content_type = 'text/markdown',  # 长描述内容类型
  url = 'https://github.com/lucidrains/classifier-free-guidance-pytorch',  # URL
  keywords = [  # 关键词
    'artificial intelligence',
    'deep learning',
    'classifier free guidance',
    'text conditioning and guidance'
  ],
  install_requires=[  # 安装依赖
    'beartype',
    'einops>=0.7',
    'ftfy',
    'open-clip-torch>=2.8.0',
    'torch>=2.0',
    'transformers[torch]'
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