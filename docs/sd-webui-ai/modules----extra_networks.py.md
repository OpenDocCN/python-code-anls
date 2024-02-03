# `stable-diffusion-webui\modules\extra_networks.py`

```
# 导入所需的模块
import json
import os
import re
import logging
from collections import defaultdict

# 定义额外网络的注册表和别名字典
extra_network_registry = {}
extra_network_aliases = {}

# 初始化额外网络注册表和别名字典
def initialize():
    extra_network_registry.clear()
    extra_network_aliases.clear()

# 注册额外网络到注册表
def register_extra_network(extra_network):
    extra_network_registry[extra_network.name] = extra_network

# 注册额外网络别名到别名字典
def register_extra_network_alias(extra_network, alias):
    extra_network_aliases[alias] = extra_network

# 注册默认的额外网络
def register_default_extra_networks():
    # 导入额外网络模块并注册到注册表
    from modules.extra_networks_hypernet import ExtraNetworkHypernet
    register_extra_network(ExtraNetworkHypernet())

# 定义额外网络参数类
class ExtraNetworkParams:
    def __init__(self, items=None):
        self.items = items or []
        self.positional = []
        self.named = {}

        # 遍历参数列表，将参数分为位置参数和命名参数
        for item in self.items:
            parts = item.split('=', 2) if isinstance(item, str) else [item]
            if len(parts) == 2:
                self.named[parts[0]] = parts[1]
            else:
                self.positional.append(item)

    def __eq__(self, other):
        return self.items == other.items

# 定义额外网络类
class ExtraNetwork:
    def __init__(self, name):
        self.name = name
    # 激活额外网络，处理每次运行。在这里激活额外网络应该做的事情。
    # 在 params_list 中传递与这个额外网络相关的参数。
    # 用户通过在提示中指定这个来传递参数:

    # <name:arg1:arg2:arg3>

    # 其中 name 与 ExtraNetwork 对象的名称匹配，arg1:arg2:arg3 是由冒号分隔的任意数量的文本参数。

    # 即使用户在提示中没有提及这个 ExtraNetwork，调用仍会进行，params_list 为空 -
    # 在这种情况下，这个额外网络的所有效果应该被禁用。

    # 可以在 deactivate() 之前被多次调用 - 每次新调用应该完全覆盖之前的调用。

    # 例如，如果这个 ExtraNetwork 的名称是 'hypernet'，用户的提示是:

    # > "1girl, <hypernet:agm:1.1> <extrasupernet:master:12:13:14> <hypernet:ray>"

    # params_list 将是:

    # [
    #     ExtraNetworkParams(items=["agm", "1.1"]),
    #     ExtraNetworkParams(items=["ray"])
    # ]

    def activate(self, p, params_list):
        # 抛出未实现错误，需要在子类中实现
        raise NotImplementedError

    # 停用额外网络，用于处理结束时的清理工作。这里不需要做任何事情。
    def deactivate(self, p):
        # 抛出未实现错误，需要在子类中实现
        raise NotImplementedError
# 查找额外网络的参数并返回一个字典，将 ExtraNetwork 对象映射到这些额外网络参数的列表
def lookup_extra_networks(extra_network_data):
    # 初始化结果字典
    res = {}

    # 遍历额外网络数据字典的键值对
    for extra_network_name, extra_network_args in list(extra_network_data.items()):
        # 获取额外网络对象或者别名
        extra_network = extra_network_registry.get(extra_network_name, None)
        alias = extra_network_aliases.get(extra_network_name, None)

        # 如果别名存在且额外网络对象不存在，则使用别名
        if alias is not None and extra_network is None:
            extra_network = alias

        # 如果额外网络对象不存在，则记录日志并继续下一个循环
        if extra_network is None:
            logging.info(f"Skipping unknown extra network: {extra_network_name}")
            continue

        # 将额外网络对象和参数列表添加到结果字典中
        res.setdefault(extra_network, []).extend(extra_network_args)

    # 返回结果字典
    return res


def activate(p, extra_network_data):
    """调用指定顺序中额外网络数据中的额外网络的激活函数，然后调用所有剩余注册网络的激活函数并传入空参数列表"""

    # 已激活的网络列表
    activated = []
    # 遍历额外网络数据字典，获取额外网络和其参数
    for extra_network, extra_network_args in lookup_extra_networks(extra_network_data).items():
        # 尝试激活额外网络，并将激活成功的额外网络添加到已激活列表中
        try:
            extra_network.activate(p, extra_network_args)
            activated.append(extra_network)
        # 捕获激活额外网络时可能出现的异常，并显示错误信息
        except Exception as e:
            errors.display(e, f"activating extra network {extra_network.name} with arguments {extra_network_args}")

    # 遍历额外网络注册表中的额外网络
    for extra_network_name, extra_network in extra_network_registry.items():
        # 如果额外网络已经在已激活列表中，则跳过
        if extra_network in activated:
            continue

        # 尝试激活额外网络，并显示错误信息
        try:
            extra_network.activate(p, [])
        except Exception as e:
            errors.display(e, f"activating extra network {extra_network_name}")

    # 如果存在脚本对象，则在激活额外网络后执行脚本
    if p.scripts is not None:
        p.scripts.after_extra_networks_activate(p, batch_number=p.iteration, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds, extra_network_data=extra_network_data)
# 为指定的网络数据和额外网络数据中的额外网络按指定顺序调用deactivate，然后调用所有剩余注册网络的deactivate
def deactivate(p, extra_network_data):
    # 查找额外网络数据中的额外网络
    data = lookup_extra_networks(extra_network_data)

    # 遍历额外网络数据中的额外网络
    for extra_network in data:
        try:
            # 调用额外网络的deactivate方法
            extra_network.deactivate(p)
        except Exception as e:
            # 显示错误信息
            errors.display(e, f"deactivating extra network {extra_network.name}")

    # 遍历额外网络注册表中的额外网络
    for extra_network_name, extra_network in extra_network_registry.items():
        # 如果额外网络在数据中，则跳过
        if extra_network in data:
            continue

        try:
            # 调用额外网络的deactivate方法
            extra_network.deactivate(p)
        except Exception as e:
            # 显示错误信息
            errors.display(e, f"deactivating unmentioned extra network {extra_network_name}")


# 编译正则表达式，用于匹配额外网络
re_extra_net = re.compile(r"<(\w+):([^>]+)>")


# 解析提示信息中的额外网络参数
def parse_prompt(prompt):
    res = defaultdict(list)

    # 定义匹配到额外网络参数时的处理函数
    def found(m):
        name = m.group(1)
        args = m.group(2)

        # 将额外网络参数添加到结果字典中
        res[name].append(ExtraNetworkParams(items=args.split(":"))

        return ""

    # 使用正则表达式替换提示信息中的额外网络参数
    prompt = re.sub(re_extra_net, found, prompt)

    return prompt, res


# 解析多个提示信息中的额外网络参数
def parse_prompts(prompts):
    res = []
    extra_data = None

    # 遍历多个提示信息
    for prompt in prompts:
        # 解析提示信息中的额外网络参数
        updated_prompt, parsed_extra_data = parse_prompt(prompt)

        if extra_data is None:
            extra_data = parsed_extra_data

        res.append(updated_prompt)

    return res, extra_data


# 获取用户元数据
def get_user_metadata(filename):
    if filename is None:
        return {}

    # 获取文件名和扩展名
    basename, ext = os.path.splitext(filename)
    metadata_filename = basename + '.json'

    metadata = {}
    try:
        # 如果元数据文件存在，则读取其中的内容
        if os.path.isfile(metadata_filename):
            with open(metadata_filename, "r", encoding="utf8") as file:
                metadata = json.load(file)
    except Exception as e:
        # 显示错误信息
        errors.display(e, f"reading extra network user metadata from {metadata_filename}")

    return metadata
```