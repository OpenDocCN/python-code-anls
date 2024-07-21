# `.\pytorch\test\load_torchscript_model.py`

```py
# 导入 sys 模块，用于处理命令行参数和退出程序
import sys

# 导入 PyTorch 库，进行模型加载和数据处理
import torch

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 使用 torch.jit.load() 加载输入的脚本模型文件
    script_mod = torch.jit.load(sys.argv[1])
    # 使用 torch.load() 加载输入的原始模型文件（在文件名后加上 ".orig" 后缀）
    mod = torch.load(sys.argv[1] + ".orig")
    # 打印加载的脚本模型对象
    print(script_mod)
    # 生成一个随机输入张量，大小为 (2, 28 * 28)，用于模型推理
    inp = torch.rand(2, 28 * 28)
    # 调用加载的原始模型进行推理，忽略返回结果
    _ = mod(inp)
    # 退出程序，返回状态码 0 表示正常退出
    sys.exit(0)
```