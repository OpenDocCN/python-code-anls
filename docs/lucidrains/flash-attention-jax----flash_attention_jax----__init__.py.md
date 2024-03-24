# `.\lucidrains\flash-attention-jax\flash_attention_jax\__init__.py`

```py
# 从 flash_attention_jax 模块中导入 flash_attention 函数
from flash_attention_jax.flash_attention import flash_attention
# 从 flash_attention_jax 模块中导入 cosine_sim_flash_attention 函数
from flash_attention_jax.cosine_sim_flash_attention import cosine_sim_flash_attention
# 从 flash_attention_jax 模块中导入 causal_flash_attention 函数
from flash_attention_jax.causal_flash_attention import causal_flash_attention
# 从 flash_attention_jax 模块中导入 rabe_attention 函数
from flash_attention_jax.rabe_attention import rabe_attention
# 从 flash_attention_jax 模块中导入 attention, causal_attention, cosine_sim_attention 函数
from flash_attention_jax.attention import attention, causal_attention, cosine_sim_attention

# 从 flash_attention_jax.utils 模块中导入 value_and_grad_difference, PRNGKeyGenerator 函数
from flash_attention_jax.utils import value_and_grad_difference, PRNGKeyGenerator

# 将 attention 函数赋值给 plain_attention 变量
plain_attention = attention
```