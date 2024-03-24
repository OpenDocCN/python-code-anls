# `.\lucidrains\flash-cosine-sim-attention\flash_cosine_sim_attention\version.py`

```
# 定义当前模块的版本号
__version__ = '0.1.40'

# 根据当前模块的版本号生成 CUDA 包的名称
__cuda_pkg_name__ = f'flash_cosine_sim_attention_cuda_{__version__.replace(".", "_")}'
```