# `stable-diffusion-webui\modules\hypernetworks\ui.py`

```py
# 导入必要的模块
import html
import gradio as gr
import modules.hypernetworks.hypernetwork
from modules import devices, sd_hijack, shared

# 定义不可用的激活函数列表和可用的激活函数列表
not_available = ["hardswish", "multiheadattention"]
keys = [x for x in modules.hypernetworks.hypernetwork.HypernetworkModule.activation_dict if x not in not_available]

# 创建超网络
def create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, dropout_structure=None):
    # 调用创建超网络的函数
    filename = modules.hypernetworks.hypernetwork.create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure, activation_func, weight_init, add_layer_norm, use_dropout, dropout_structure)
    
    # 更新下拉框选项
    return gr.Dropdown.update(choices=sorted(shared.hypernetworks)), f"Created: {filename}", ""

# 训练超网络
def train_hypernetwork(*args):
    # 清空已加载的超网络
    shared.loaded_hypernetworks = []
    
    # 断言不使用低显存训练模型
    assert not shared.cmd_opts.lowvram, 'Training models with lowvram is not possible'
    
    try:
        # 恢复优化
        sd_hijack.undo_optimizations()
        
        # 训练超网络
        hypernetwork, filename = modules.hypernetworks.hypernetwork.train_hypernetwork(*args)
        
        # 返回训练结果信息
        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {hypernetwork.step} steps.
Hypernetwork saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        # 将模型转移到指定设备
        shared.sd_model.cond_stage_model.to(devices.device)
        shared.sd_model.first_stage_model.to(devices.device)
        # 应用优化
        sd_hijack.apply_optimizations()
```