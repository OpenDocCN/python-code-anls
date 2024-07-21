# `.\pytorch\benchmarks\dynamo\benchmarks.py`

```
#!/usr/bin/env python3
# 导入必要的模块
import argparse  # 用于解析命令行参数的模块
import os  # 提供与操作系统交互的功能
import sys  # 提供与 Python 解释器交互的功能

from typing import Set  # 导入 Set 类型，用于声明返回值类型


# Note - hf and timm have their own version of this, torchbench does not
# TOOD(voz): Someday, consolidate all the files into one runner instead of a shim like this...
# 定义函数 model_names，用于从文件中读取模型名称并返回为集合
def model_names(filename: str) -> Set[str]:
    names = set()
    with open(filename) as fh:
        lines = fh.readlines()
        lines = [line.rstrip() for line in lines]  # 去除每行末尾的换行符
        for line in lines:
            line_parts = line.split(" ")  # 根据空格分割每行内容
            if len(line_parts) == 1:
                line_parts = line.split(",")  # 如果分割结果为单个元素，则再按逗号分割
            model_name = line_parts[0]  # 取分割后的第一个元素作为模型名称
            names.add(model_name)  # 将模型名称添加到集合中
    return names  # 返回模型名称集合


# 从文件中读取不同模型的名称集合
TIMM_MODEL_NAMES = model_names(
    os.path.join(os.path.dirname(__file__), "timm_models_list.txt")
)
HF_MODELS_FILE_NAME = model_names(
    os.path.join(os.path.dirname(__file__), "huggingface_models_list.txt")
)
TORCHBENCH_MODELS_FILE_NAME = model_names(
    os.path.join(os.path.dirname(__file__), "all_torchbench_models_list.txt")
)

# 检查不同模型名称集合之间是否没有交集
assert TIMM_MODEL_NAMES.isdisjoint(HF_MODELS_FILE_NAME)
assert TIMM_MODEL_NAMES.isdisjoint(TORCHBENCH_MODELS_FILE_NAME)
assert TORCHBENCH_MODELS_FILE_NAME.isdisjoint(HF_MODELS_FILE_NAME)


# 定义命令行参数解析函数
def parse_args(args=None):
    parser = argparse.ArgumentParser()  # 创建 ArgumentParser 对象
    parser.add_argument(
        "--only",
        help="""Run just one model from whichever model suite it belongs to. Or
        specify the path and class name of the model in format like:
        --only=path:<MODEL_FILE_PATH>,class:<CLASS_NAME>

        Due to the fact that dynamo changes current working directory,
        the path should be an absolute path.

        The class should have a method get_example_inputs to return the inputs
        for the model. An example looks like
        ```
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

            def get_example_inputs(self):
                return (torch.randn(2, 10),)
        ```
    """,
    )
    return parser.parse_known_args(args)  # 解析命令行参数并返回


if __name__ == "__main__":
    args, unknown = parse_args()  # 解析命令行参数
    if args.only:
        name = args.only
        if name in TIMM_MODEL_NAMES:  # 如果指定模型名称在 TIMM 模型集合中
            import timm_models  # 导入 timm_models 模块

            timm_models.timm_main()  # 调用 timm_models 模块的主函数
        elif name in HF_MODELS_FILE_NAME:  # 如果指定模型名称在 HF 模型集合中
            import huggingface  # 导入 huggingface 模块

            huggingface.huggingface_main()  # 调用 huggingface 模块的主函数
        elif name in TORCHBENCH_MODELS_FILE_NAME:  # 如果指定模型名称在 Torchbench 模型集合中
            import torchbench  # 导入 torchbench 模块

            torchbench.torchbench_main()  # 调用 torchbench 模块的主函数
        else:
            print(f"Illegal model name? {name}")  # 输出非法模型名称
            sys.exit(-1)  # 退出程序，返回错误状态码
    else:
        import torchbench  # 导入 torchbench 模块

        torchbench.torchbench_main()  # 调用 torchbench 模块的主函数

        import huggingface  # 导入 huggingface 模块

        huggingface.huggingface_main()  # 调用 huggingface 模块的主函数

        import timm_models  # 导入 timm_models 模块

        timm_models.timm_main()  # 调用 timm_models 模块的主函数
```