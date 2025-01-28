# `.\minimind-v\export_model.py`

```py
# 导入 PyTorch 库
import torch
# 导入警告模块
import warnings
# 从 transformers 库导入 AutoTokenizer 和 AutoModelForCausalLM 类
from transformers import AutoTokenizer, AutoModelForCausalLM
# 从 model.LMConfig 模块导入 LMConfig 类
from model.LMConfig import LMConfig
# 从 model.model 模块导入 Transformer 类
from model.model import Transformer

# 忽略 UserWarning 类的警告信息
warnings.filterwarnings('ignore', category=UserWarning)

# 定义一个函数，用于统计模型中所有可训练参数的数量
def count_parameters(model):
    # 遍历模型的所有参数，计算需要梯度更新的参数总数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 定义一个函数，用于导出 transformer 模型
def export_transformers_model():
    # 从指定路径加载分词器，设置 trust_remote_code 为 True，使用慢速 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer',
                                              trust_remote_code=True, use_fast=False)
    # 注册 LMConfig 类以便于自动加载
    LMConfig.register_for_auto_class()
    # 注册 Transformer 类以便于自动加载
    Transformer.register_for_auto_class("AutoModelForCausalLM")

    # 创建一个 LMConfig 的实例，包含模型的配置信息
    lm_config = LMConfig()
    # 使用配置文件实例化一个 Transformer 模型
    lm_model = Transformer(lm_config)
    # 检查是否有 GPU 可用，选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 根据配置文件中是否使用 Moe 技术，设置模型检查点路径
    moe_path = '_moe' if lm_config.use_moe else ''
    # 构造最终的模型检查点路径
    ckpt_path = f'./out/{lm_config.dim}{moe_path}_vlm_sft.pth'

    # 加载模型检查点的状态字典，将其加载到当前设备
    state_dict = torch.load(ckpt_path, map_location=device)
    # 设置不需要的前缀，准备移除状态字典中的对应前缀
    unwanted_prefix = '_orig_mod.'
    # 遍历状态字典的所有项，移除不需要的前缀
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # 将状态字典加载到模型中，并允许部分参数不严格匹配
    lm_model.load_state_dict(state_dict, strict=False)
    # 打印模型的参数数量，以百万和十亿为单位
    print(f'模型参数: {count_parameters(lm_model) / 1e6} 百万 = {count_parameters(lm_model) / 1e9} B (Billion)')

    # 保存模型和分词器到指定的路径
    lm_model.save_pretrained("minimind-v-v1-small", safe_serialization=False)
    tokenizer.save_pretrained("minimind-v-v1-small")

# 如果当前脚本是主程序，则执行以下操作
if __name__ == '__main__':
    # 调用函数导出 transformer 模型
    export_transformers_model()
```