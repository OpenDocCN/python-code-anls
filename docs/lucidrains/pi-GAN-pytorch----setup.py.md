# `.\lucidrains\pi-GAN-pytorch\setup.py`

```
# 导入设置安装和查找包的函数
from setuptools import setup, find_packages

# 设置包的信息
setup(
  # 包的名称
  name = 'pi-gan-pytorch',
  # 查找所有包
  packages = find_packages(),
  # 版本号
  version = '0.0.11',
  # 许可证
  license='MIT',
  # 描述
  description = 'π-GAN - Pytorch',
  # 作者
  author = 'Phil Wang',
  # 作者邮箱
  author_email = 'lucidrains@gmail.com',
  # 项目链接
  url = 'https://github.com/lucidrains/pi-gan-pytorch',
  # 关键词
  keywords = [
    'artificial intelligence',
    'generative adversarial network'
  ],
  # 安装依赖
  install_requires=[
    'einops>=0.3',
    'pillow',
    'torch>=1.6',
    'torchvision',
    'tqdm'
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