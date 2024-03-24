# `.\lucidrains\scattering-compositional-learner\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 设置包的元信息
setup(
  name = 'scattering-transform',  # 包名
  packages = find_packages(),  # 查找所有包
  version = '0.0.7',  # 版本号
  license='MIT',  # 许可证
  description = 'Scattering Transform module from the paper Scattering Compositional Learner',  # 描述
  author = 'Phil Wang',  # 作者
  author_email = 'lucidrains@gmail.com',  # 作者邮箱
  url = 'https://github.com/lucidrains/scattering-compositional-learner',  # 项目链接
  keywords = ['artificial intelligence', 'deep learning', 'reasoning'],  # 关键词
  install_requires=[
      'torch'  # 安装依赖
  ],
  classifiers=[
      'Development Status :: 4 - Beta',  # 开发状态
      'Intended Audience :: Developers',  # 目标受众
      'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 主题
      'License :: OSI Approved :: MIT License',  # 许可证类型
      'Programming Language :: Python :: 3.6',  # 编程语言
  ],
)
```