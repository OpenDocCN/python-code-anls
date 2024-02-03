# `stable-diffusion-webui\modules\sd_hijack_ip2p.py`

```py
# 导入 os.path 模块
import os.path

# 判断是否应该劫持 IP2P
def should_hijack_ip2p(checkpoint_info):
    # 从 modules 模块中导入 sd_models_config 模块
    from modules import sd_models_config

    # 获取检查点文件名的基本名称，并转换为小写
    ckpt_basename = os.path.basename(checkpoint_info.filename).lower()
    # 获取与检查点文件名最接近的配置文件的基本名称，并转换为小写
    cfg_basename = os.path.basename(sd_models_config.find_checkpoint_config_near_filename(checkpoint_info)).lower()

    # 检查检查点文件名中是否包含 "pix2pix"，并且配置文件名中不包含 "pix2pix"
    return "pix2pix" in ckpt_basename and "pix2pix" not in cfg_basename
```