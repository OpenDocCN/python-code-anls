# `transformer_vq\setup.py`

```
# 导入所需的模块
from io import open  # 导入 open 函数
from setuptools import find_packages  # 导入 find_packages 函数
from setuptools import setup  # 导入 setup 函数

# 设置包的信息
setup(
    name="transformer_vq",  # 包的名称
    version="13.0.1",  # 版本号
    url="https://github.com/transformer-vq/transformer_vq/",  # 项目的 URL
    license="MIT",  # 许可证
    author="Anonymous Authors; Paper and Code Under Double-Blind Review at ICLR 2024",  # 作者信息
    description="Official Transformer-VQ implementation in Jax.",  # 包的描述
    long_description=open("README.md", encoding="utf-8").read(),  # 读取 README.md 文件作为长描述
    packages=find_packages(where="src"),  # 查找包的位置
    package_dir={"": "src"},  # 包的目录
    platforms="any",  # 支持的平台
    python_requires=">=3.8",  # 需要的 Python 版本
    install_requires=[
        "chex>=0.1.7",  # 安装所需的依赖
        "datasets>=2.11.0",  # 安装 datasets 包，版本需大于等于 2.11.0
        "jaxlib==0.4.9",  # 安装 jaxlib 包，版本为 0.4.9
        "flax==0.6.11",  # 安装 flax 包，版本为 0.6.11
        "numpy>=1.22.0",  # 安装 numpy 包，版本需大于等于 1.22.0
        "optax==0.1.5",  # 安装 optax 包，版本为 0.1.5
        "orbax-checkpoint==0.1.7",  # 安装 orbax-checkpoint 包，版本为 0.1.7
        "requests>=2.28.1",  # 安装 requests 包，版本需大于等于 2.28.1
        "sentencepiece==0.1.96",  # 安装 sentencepiece 包，版本为 0.1.96
        "seqio==0.0.16",  # 安装 seqio 包，版本为 0.0.16
        "tensorflow==2.12.1",  # 安装 tensorflow 包，版本为 2.12.1
        "tensorflow-text==2.12.1",  # 安装 tensorflow-text 包，版本为 2.12.1
        "tensorflow-datasets>=4.9.1",  # 安装 tensorflow-datasets 包，版本需大于等于 4.9.1
        "tensorboard>=2.10.1",  # 安装 tensorboard 包，版本需大于等于 2.10.1
        "tensorstore>=0.1.35",  # 安装 tensorstore 包，版本需大于等于 0.1.35
        "tqdm>=4.65.0",  # 安装 tqdm 包，版本需大于等于 4.65.0
        "wandb<0.15.0",  # 安装 wandb 包，版本需小于 0.15.0
    ],
    extras_require={
        "dev": [
            "pre-commit",  # 安装 pre-commit 包，用于开发环境
# 定义一个字典，包含不同环境下需要安装的软件包
{
    "common": [  # 在所有环境下都需要安装的软件包
        "numpy==1.21.2",
        "scipy==1.7.1",
        "scikit-learn==0.24.2",
    ],
    "dev": [  # 开发环境下需要安装的软件包
        "pytest",
        "pytest-cov",
    ],
    "no_tpu": [  # 没有 TPU 环境下需要安装的软件包
        "jax==0.4.9",
    ],
    "tpu": [  # TPU 环境下需要安装的软件包
        "jax[tpu]==0.4.9",
        "protobuf<=3.20.1",
    ],
    "viz": [  # 可视化环境下需要安装的软件包
        "matplotlib==3.5.3",
        "pandas==1.4.4",
        "seaborn==0.12.1",
    ],
}
```