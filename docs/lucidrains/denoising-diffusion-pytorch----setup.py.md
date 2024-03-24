# `.\lucidrains\denoising-diffusion-pytorch\setup.py`

```
# 导入设置工具和查找包工具
from setuptools import setup, find_packages

# 执行版本文件中的代码，将版本信息导入当前环境
exec(open('denoising_diffusion_pytorch/version.py').read())

# 设置包的元数据
setup(
  name = 'denoising-diffusion-pytorch', # 包名
  packages = find_packages(), # 查找所有包
  version = __version__, # 使用之前导入的版本信息
  license='MIT', # 许可证
  description = 'Denoising Diffusion Probabilistic Models - Pytorch', # 描述
  author = 'Phil Wang', # 作者
  author_email = 'lucidrains@gmail.com', # 作者邮箱
  url = 'https://github.com/lucidrains/denoising-diffusion-pytorch', # 项目链接
  long_description_content_type = 'text/markdown', # 长描述内容类型
  keywords = [
    'artificial intelligence', # 关键词
    'generative models'
  ],
  install_requires=[ # 安装依赖
    'accelerate',
    'einops',
    'ema-pytorch>=0.4.2',
    'numpy',
    'pillow',
    'pytorch-fid',
    'torch',
    'torchvision',
    'tqdm'
  ],
  classifiers=[ # 分类器
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
```