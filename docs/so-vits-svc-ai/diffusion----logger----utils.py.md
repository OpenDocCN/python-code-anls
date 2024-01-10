# `so-vits-svc\diffusion\logger\utils.py`

```
# 导入 json、os、torch 和 yaml 模块
import json
import os
import torch
import yaml

# 定义函数 traverse_dir，用于遍历目录
def traverse_dir(
        root_dir,  # 根目录
        extensions,  # 文件扩展名列表
        amount=None,  # 遍历文件数量限制
        str_include=None,  # 包含指定字符串
        str_exclude=None,  # 排除指定字符串
        is_pure=False,  # 是否返回纯净路径
        is_sort=False,  # 是否排序
        is_ext=True):  # 是否包含文件扩展名

    # 初始化文件列表和计数器
    file_list = []
    cnt = 0
    # 遍历根目录及其子目录
    for root, _, files in os.walk(root_dir):
        for file in files:
            # 判断文件是否以指定扩展名结尾
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # 拼接文件路径
                mix_path = os.path.join(root, file)
                # 获取纯净路径
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # 判断是否达到遍历文件数量限制
                if (amount is not None) and (cnt == amount):
                    # 如果需要排序，则对文件列表进行排序后返回
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # 检查是否包含指定字符串或排除指定字符串
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                # 如果不包含文件扩展名，则去除扩展名
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                # 将路径添加到文件列表中
                file_list.append(pure_path)
                cnt += 1
    # 如果需要排序，则对文件列表进行排序后返回
    if is_sort:
        file_list.sort()
    return file_list

# 定义 DotDict 类，继承自 dict 类
class DotDict(dict):
    # 获取属性
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    # 设置属性
    __setattr__ = dict.__setitem__    
    # 删除属性
    __delattr__ = dict.__delitem__

# 定义函数 get_network_paras_amount，用于获取模型参数数量
def get_network_paras_amount(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        # 计算可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 将模型名称和可训练参数数量添加到字典中
        info[model_name] = trainable_params
    return info

# 定义函数 load_config，用于加载配置文件
def load_config(path_config):
    # 使用只读模式打开配置文件
    with open(path_config, "r") as config:
        # 从配置文件中加载安全的 YAML 数据并赋值给args
        args = yaml.safe_load(config)
    # 将args转换为DotDict对象
    args = DotDict(args)
    # 返回args对象
    return args
# 保存配置信息到指定路径的文件中
def save_config(path_config,config):
    # 将配置信息转换为字典类型
    config = dict(config)
    # 打开指定路径的文件，以写入模式
    with open(path_config, "w") as f:
        # 将配置信息字典转换为 YAML 格式并写入文件
        yaml.dump(config, f)

# 将模型参数保存为 JSON 文件
def to_json(path_params, path_json):
    # 加载模型参数
    params = torch.load(path_params, map_location=torch.device('cpu'))
    raw_state_dict = {}
    # 遍历模型参数，将参数值展平并转换为 numpy 数组后保存到字典中
    for k, v in params.items():
        val = v.flatten().numpy().tolist()
        raw_state_dict[k] = val
    # 将字典中的参数值保存为 JSON 格式到指定路径的文件中，缩进为制表符
    with open(path_json, 'w') as outfile:
        json.dump(raw_state_dict, outfile, indent= "\t")

# 将张量转换为 numpy 数组
def convert_tensor_to_numpy(tensor, is_squeeze=True):
    # 如果需要挤压张量，则进行挤压操作
    if is_squeeze:
        tensor = tensor.squeeze()
    # 如果张量需要梯度，则将其分离出计算图
    if tensor.requires_grad:
        tensor = tensor.detach()
    # 如果张量在 GPU 上，则将其移动到 CPU 上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # 返回张量对应的 numpy 数组
    return tensor.numpy()

# 加载模型及其参数
def load_model(
        expdir, 
        model,
        optimizer,
        name='model',
        postfix='',
        device='cpu'):
    # 如果未指定后缀，则使用默认后缀
    if postfix == '':
        postfix = '_' + postfix
    # 拼接模型路径
    path = os.path.join(expdir, name+postfix)
    # 遍历指定目录下的所有 .pt 文件
    path_pt = traverse_dir(expdir, ['pt'], is_ext=False)
    global_step = 0
    # 如果存在 .pt 文件
    if len(path_pt) > 0:
        # 获取所有 .pt 文件的后缀数字
        steps = [s[len(path):] for s in path_pt]
        # 获取最大的后缀数字
        maxstep = max([int(s) if s.isdigit() else 0 for s in steps])
        # 如果最大后缀数字大于等于 0
        if maxstep >= 0:
            path_pt = path+str(maxstep)+'.pt'
        else:
            path_pt = path+'best.pt'
        # 打印信息，恢复模型参数
        print(' [*] restoring model from', path_pt)
        # 加载模型参数
        ckpt = torch.load(path_pt, map_location=torch.device(device))
        global_step = ckpt['global_step']
        # 加载模型参数
        model.load_state_dict(ckpt['model'], strict=False)
        # 如果存在优化器参数，则加载优化器参数
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
    # 返回全局步数、模型及优化器
    return global_step, model, optimizer
```