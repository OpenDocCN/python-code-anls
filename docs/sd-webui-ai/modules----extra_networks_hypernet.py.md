# `stable-diffusion-webui\modules\extra_networks_hypernet.py`

```
# 从 modules 模块中导入 extra_networks 和 shared 模块
# 从 modules.hypernetworks 模块中导入 hypernetwork 模块
from modules import extra_networks, shared
from modules.hypernetworks import hypernetwork

# 创建 ExtraNetworkHypernet 类，继承自 ExtraNetwork 类
class ExtraNetworkHypernet(extra_networks.ExtraNetwork):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法，传入参数 'hypernet'
        super().__init__('hypernet')

    # 激活方法，接受参数 p 和 params_list
    def activate(self, p, params_list):
        # 获取 shared 模块中的 sd_hypernetwork 变量
        additional = shared.opts.sd_hypernetwork

        # 如果 additional 不为 "None"，且在 shared.hypernetworks 中，且 params_list 中不存在 additional
        if additional != "None" and additional in shared.hypernetworks and not any(x for x in params_list if x.items[0] == additional):
            # 构建 hypernet_prompt_text 字符串
            hypernet_prompt_text = f"<hypernet:{additional}:{shared.opts.extra_networks_default_multiplier}>"
            # 将 hypernet_prompt_text 添加到所有提示中
            p.all_prompts = [f"{prompt}{hypernet_prompt_text}" for prompt in p.all_prompts]
            # 将包含 additional 和 shared.opts.extra_networks_default_multiplier 的 ExtraNetworkParams 对象添加到 params_list 中
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        # 初始化 names 和 multipliers 列表
        names = []
        multipliers = []
        # 遍历 params_list 中的参数
        for params in params_list:
            # 断言 params.items 不为空
            assert params.items

            # 将 params.items[0] 添加到 names 列表中
            names.append(params.items[0])
            # 如果 params.items 的长度大于 1，则将 params.items[1] 转换为浮点数并添加到 multipliers 列表中，否则添加 1.0
            multipliers.append(float(params.items[1]) if len(params.items) > 1 else 1.0)

        # 调用 hypernetwork 模块的 load_hypernetworks 方法，传入 names 和 multipliers
        hypernetwork.load_hypernetworks(names, multipliers)

    # 停用方法，接受参数 p
    def deactivate(self, p):
        # 空方法，不执行任何操作
        pass
```