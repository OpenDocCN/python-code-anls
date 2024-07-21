# `.\pytorch\docs\source\scripts\onnx\build_onnx_dynamo_diagnostics_rules_md.py`

```py
# 导入 argparse 模块，用于命令行参数解析
import argparse
# 导入 os 模块，用于操作文件系统路径
import os
# 从 dataclasses 模块中导入 fields 函数，用于获取数据类的字段信息
from dataclasses import fields
# 从 torch.onnx._internal.diagnostics 模块中导入 infra，用于诊断工具的基础设施
from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra


# 定义函数 gen_docs，用于生成诊断规则的文档
def gen_docs(out_dir: str):
    # 创建输出目录，如果目录已存在则不做任何操作
    os.makedirs(out_dir, exist_ok=True)
    # 遍历 diagnostics.rules 数据类的字段
    for field in fields(diagnostics.rules):
        # 获取字段名对应的规则对象
        rule = getattr(diagnostics.rules, field.name)
        # 如果规则对象不是 infra.Rule 类型，则跳过
        if not isinstance(rule, infra.Rule):
            continue
        # 如果规则对象的 id 不以 "FXE" 开头，则跳过
        if not rule.id.startswith("FXE"):
            # 只生成 dynamo_export 规则的文档，排除 TorchScript ONNX 导出器的规则
            continue
        # 构建文档的标题，格式为 "规则ID:规则名称"
        title = f"{rule.id}:{rule.name}"
        # 获取规则对象的完整描述文档（Markdown 格式）
        full_description_markdown = rule.full_description_markdown
        # 断言完整描述文档不为 None，确保有完整的 Markdown 描述
        assert (
            full_description_markdown is not None
        ), f"Expected {title} to have a full description in markdown"
        # 打开输出文件，以写入模式创建文件对象
        with open(f"{out_dir}/{title}.md", "w") as f:
            # 写入文档的标题部分，使用 Markdown 格式化
            f.write(f"# {title}\n")
            # 写入完整的描述文档内容
            f.write(full_description_markdown)


# 定义主函数 main，用于解析命令行参数并调用 gen_docs 函数
def main() -> None:
    # 创建 argparse.ArgumentParser 对象，设置程序的描述信息
    parser = argparse.ArgumentParser(
        description="Generate ONNX diagnostics rules doc in markdown."
    )
    # 添加命令行参数的定义，指定输出目录的路径
    parser.add_argument(
        "out_dir", metavar="OUT_DIR", help="path to output directory for docs"
    )
    # 解析命令行参数，将解析结果存储在 args 变量中
    args = parser.parse_args()
    # 调用 gen_docs 函数，传入输出目录路径参数
    gen_docs(args.out_dir)


# 判断当前脚本是否作为主程序运行
if __name__ == "__main__":
    # 调用主函数 main
    main()
```