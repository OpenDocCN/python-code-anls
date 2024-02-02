# `transformer_vq\setup.py`

```py
# 导入所需的模块
from io import open
from setuptools import find_packages
from setuptools import setup

# 设置包的元信息
setup(
    name="transformer_vq",  # 包名
    version="13.0.1",  # 版本号
    url="https://github.com/transformer-vq/transformer_vq/",  # 项目地址
    license="MIT",  # 许可证
    author="Anonymous Authors; Paper and Code Under Double-Blind Review at ICLR 2024",  # 作者信息
    description="Official Transformer-VQ implementation in Jax.",  # 描述
    long_description=open("README.md", encoding="utf-8").read(),  # 详细描述，从 README.md 文件中读取
    packages=find_packages(where="src"),  # 查找包的位置
    package_dir={"": "src"},  # 包的目录结构
    platforms="any",  # 支持的平台
    python_requires=">=3.8",  # Python 版本要求
    install_requires=[  # 安装依赖的包
        "chex>=0.1.7",
        "datasets>=2.11.0",
        "jaxlib==0.4.9",
        "flax==0.6.11",
        "numpy>=1.22.0",
        "optax==0.1.5",
        "orbax-checkpoint==0.1.7",
        "requests>=2.28.1",
        "sentencepiece==0.1.96",
        "seqio==0.0.16",
        "tensorflow==2.12.1",
        "tensorflow-text==2.12.1",
        "tensorflow-datasets>=4.9.1",
        "tensorboard>=2.10.1",
        "tensorstore>=0.1.35",
        "tqdm>=4.65.0",
        "wandb<0.15.0",
    ],
    extras_require={  # 额外的依赖
        "dev": [  # 开发环境依赖
            "pre-commit",
            "pytest",
            "pytest-cov",
        ],
        "no_tpu": [  # 不使用 TPU 时的依赖
            "jax==0.4.9",
        ],
        "tpu": [  # 使用 TPU 时的依赖
            "jax[tpu]==0.4.9",
            "protobuf<=3.20.1",
        ],
        "viz": [  # 可视化相关的依赖
            "matplotlib==3.5.3",
            "pandas==1.4.4",
            "seaborn==0.12.1",
        ],
    },
)
```