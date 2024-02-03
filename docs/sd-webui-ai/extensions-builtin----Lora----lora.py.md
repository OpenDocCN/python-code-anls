# `stable-diffusion-webui\extensions-builtin\Lora\lora.py`

```
# 导入 networks 模块
import networks

# 获取可用的 LoRa 网络列表
list_available_loras = networks.list_available_networks

# 获取可用的 LoRa 网络
available_loras = networks.available_networks

# 获取可用的 LoRa 网络别名
available_lora_aliases = networks.available_network_aliases

# 获取可用的 LoRa 网络哈希查找表
available_lora_hash_lookup = networks.available_network_hash_lookup

# 获取禁止使用的 LoRa 网络别名列表
forbidden_lora_aliases = networks.forbidden_network_aliases

# 获取已加载的 LoRa 网络列表
loaded_loras = networks.loaded_networks
```