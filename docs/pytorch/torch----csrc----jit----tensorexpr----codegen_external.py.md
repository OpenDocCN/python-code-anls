# `.\pytorch\torch\csrc\jit\tensorexpr\codegen_external.py`

```py
# 忽略类型检查错误，这是为了在运行时不对类型进行强制检查
# 导入 argparse 库，用于解析命令行参数
import argparse

# 导入 torchgen 库中的 model 模块
import torchgen.model as model
# 从 torchgen.gen 中导入 FileManager 和 parse_native_yaml 函数
from torchgen.gen import FileManager, parse_native_yaml


# 定义函数 num_leading_spaces，用于计算字符串行的前导空格数
def num_leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip())


# 定义函数 deindent，用于去除代码的整体缩进
def deindent(code: str) -> str:
    lines = code.split("\n")
    min_leading_spaces = min(map(num_leading_spaces, lines))
    lines = [line[min_leading_spaces:] for line in lines]
    return "\n".join(lines)


# 定义函数 gen_external，用于生成外部函数
def gen_external(native_functions_path, tags_path, external_path):
    # 解析本地 YAML 文件，获取本地函数信息
    native_functions = parse_native_yaml(native_functions_path, tags_path)
    func_decls = []  # 存储生成的函数声明代码
    func_registrations = []  # 存储生成的函数注册代码
    for func in native_functions:
        schema = func.func
        name = schema.name.name.base  # 获取函数名称
        args = schema.arguments  # 获取函数参数信息

        # 仅支持带有 out 变体的函数调用
        if not schema.is_out_fn():
            continue

        # 目前不支持具有多个 out 参数的函数
        if len(args.out) > 1:
            continue

        # 目前不支持关键字参数
        if (
            len(args.pre_tensor_options_kwarg_only) > 0
            or len(args.post_tensor_options_kwarg_only) > 0
        ):
            continue

        self_arg = [args.self_arg.argument] if args.self_arg is not None else []
        args = (
            list(args.pre_self_positional) + self_arg + list(args.post_self_positional)
        )
        
        # 过滤出张量类型参数
        tensor_args = [
            arg
            for arg in args
            if isinstance(arg.type, model.BaseType)
            and arg.type.name == model.BaseTy.Tensor
        ]
        
        # 如果参数中不全是张量类型，则跳过该函数
        if len(tensor_args) != len(args):
            continue

        arg_names = [None] * len(args)

        tensor_decls = []
        for idx, arg in enumerate(tensor_args):
            s = f"const at::Tensor& {arg.name} = tensors[{idx + 1}];"
            tensor_decls.append(s)
            arg_names[idx] = arg.name
        nl = "\n"

        # 构建函数声明代码块
        func_decl = f"""\
void nnc_aten_{name}(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {{
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);
  at::Tensor& r = tensors[0];
  {nl.join(tensor_decls)}
  try {{
    at::{name}_out({', '.join(['r'] + arg_names)});
  }} catch (...) {{
  }}
}}"""
        
        # 构建函数注册代码块
        func_registration = f"""\
const static RegisterNNCExternalFunction nnc_{name}(
    "nnc_aten_{name}",
    nnc_aten_{name});"""
        
        # 将生成的函数声明和函数注册代码添加到相应列表中
        func_decls.append(func_decl)
        func_registrations.append(func_registration)

    # 初始化 FileManager 实例，用于文件管理
    fm = FileManager(install_dir=".", template_dir=".", dry_run=False)
    fm.write_with_template(
        "external_functions_codegen.cpp",  # 指定要写入的目标文件名
        external_path,                     # 指定外部路径
        lambda: {                          # 使用 lambda 表达式创建字典，作为模板参数
            "external_registrations": func_registrations,  # 将 func_registrations 添加到模板字典中的 "external_registrations" 键
            "external_functions": func_decls,              # 将 func_decls 添加到模板字典中的 "external_functions" 键
        },
    )
# 定义程序的主函数，不返回任何值
def main() -> None:
    # 创建参数解析器对象，设置程序描述信息
    parser = argparse.ArgumentParser(description="Generate annotated_fn_args script")
    
    # 添加命令行参数：--native-functions，用于指定 native_functions.yaml 文件的路径
    parser.add_argument(
        "--native-functions",
        "--native_functions",
        help="path to native_functions.yaml",
        default="../../../../aten/src/ATen/native/native_functions.yaml",
    )
    
    # 添加命令行参数：--tags，用于指定 tags.yaml 文件的路径
    parser.add_argument(
        "--tags",
        help="path to tags.yaml",
        default="../../../../aten/src/ATen/native/tags.yaml",
    )
    
    # 添加命令行参数：--template-path，用于指定 external_functions_codegen_template.cpp 文件的路径
    parser.add_argument(
        "--template-path",
        "--template_path",
        help="path to external_functions_codegen_template.cpp",
        default="../../../../tools/jit/templates/external_functions_codegen_template.cpp",
    )
    
    # 解析命令行参数，并将其存储到 args 对象中
    args = parser.parse_args()
    
    # 调用 gen_external 函数，传入解析得到的参数：native_functions 路径、tags 路径、template_path 路径
    gen_external(args.native_functions, args.tags, args.template_path)


# 如果当前脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```