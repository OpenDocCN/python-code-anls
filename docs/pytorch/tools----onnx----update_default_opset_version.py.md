# `.\pytorch\tools\onnx\update_default_opset_version.py`

```
#!/usr/bin/env python3
# 设置脚本的解释器为 Python 3，以便在不同环境中执行

"""Updates the default value of opset_version.

The current policy is that the default should be set to the
latest released version as of 18 months ago.

Usage:
Run with no arguments.
"""
# 脚本功能说明和用法说明

import argparse  # 导入用于解析命令行参数的模块
import datetime  # 导入处理日期和时间的模块
import os  # 导入操作系统相关功能的模块
import re  # 导入处理正则表达式的模块
import subprocess  # 导入执行外部命令和获取输出的模块
import sys  # 导入与 Python 解释器相关的系统功能
from pathlib import Path  # 导入处理文件路径的模块
from subprocess import DEVNULL  # 导入用于忽略命令输出的特殊对象
from typing import Any  # 导入用于类型提示的模块


def read_sub_write(path: str, prefix_pat: str, new_default: int) -> None:
    # 读取指定路径的文件内容
    with open(path, encoding="utf-8") as f:
        content_str = f.read()
    # 使用正则表达式替换文件内容中的特定模式
    content_str = re.sub(prefix_pat, rf"\g<1>{new_default}", content_str)
    # 将修改后的内容写回到文件中
    with open(path, "w", encoding="utf-8") as f:
        f.write(content_str)
    # 打印提示信息，指示文件已修改
    print("modified", path)


def main(args: Any) -> None:
    # 确定当前脚本所在的 PyTorch 根目录
    pytorch_dir = Path(__file__).parent.parent.parent.resolve()
    # 确定 ONNX 目录路径
    onnx_dir = pytorch_dir / "third_party" / "onnx"
    # 切换工作目录至 ONNX 目录
    os.chdir(onnx_dir)

    # 计算18个月前的日期
    date = datetime.datetime.now() - datetime.timedelta(days=18 * 30)
    # 获取指定日期前的最新一次提交哈希值
    onnx_commit = subprocess.check_output(
        ("git", "log", f"--until={date}", "--max-count=1", "--format=%H"),
        encoding="utf-8",
    ).strip()
    # 获取包含指定提交的所有标签
    onnx_tags = subprocess.check_output(
        ("git", "tag", "--list", f"--contains={onnx_commit}"), encoding="utf-8"
    )
    tag_tups = []
    semver_pat = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
    for tag in onnx_tags.splitlines():
        match = semver_pat.match(tag)
        if match:
            # 解析语义版本号并存储
            tag_tups.append(tuple(int(x) for x in match.groups()))

    # 从18个月前的版本中选择最小的语义版本号
    version_str = "{}.{}.{}".format(*min(tag_tups))

    # 打印所使用的 ONNX 发行版号
    print("Using ONNX release", version_str)

    # 获取当前 HEAD 的提交哈希值
    head_commit = subprocess.check_output(
        ("git", "log", "--max-count=1", "--format=%H", "HEAD"), encoding="utf-8"
    ).strip()

    new_default = None

    # 检出指定版本的 ONNX
    subprocess.check_call(
        ("git", "checkout", f"v{version_str}"), stdout=DEVNULL, stderr=DEVNULL
    )
    try:
        from onnx import helper  # type: ignore[import]

        for version in helper.VERSION_TABLE:
            # 查找指定版本在版本表中的默认 opset_version
            if version[0] == version_str:
                new_default = version[2]
                print("found new default opset_version", new_default)
                break
        if not new_default:
            # 如果找不到默认版本，则打印错误信息并退出
            sys.exit(
                f"failed to find version {version_str} in onnx.helper.VERSION_TABLE at commit {onnx_commit}"
            )
    finally:
        # 恢复到原来的 HEAD
        subprocess.check_call(
            ("git", "checkout", head_commit), stdout=DEVNULL, stderr=DEVNULL
        )

    # 切换回 PyTorch 根目录
    os.chdir(pytorch_dir)

    # 更新 torch/onnx/_constants.py 文件中的默认 opset_version
    read_sub_write(
        os.path.join("torch", "onnx", "_constants.py"),
        r"(ONNX_DEFAULT_OPSET = )\d+",
        new_default,
    )
    # 更新 torch/onnx/utils.py 文件中的默认 opset_version
    read_sub_write(
        os.path.join("torch", "onnx", "utils.py"),
        r"(opset_version \(int, default )\d+",
        new_default,
    )

    # 如果未跳过构建过程，则开始构建 PyTorch
    if not args.skip_build:
        print("Building PyTorch...")
        subprocess.check_call(
            ("python", "setup.py", "develop"),
        )
    # 打印提示信息，指示操作完成
    print("Updating operator .expect files")
    # 使用 subprocess 模块调用外部命令，执行 Python 脚本
    subprocess.check_call(
        # 执行的命令是运行 Python 解释器，指定运行的脚本路径为 'test/onnx/test_operators.py'，并传入参数 '--accept'
        ("python", os.path.join("test", "onnx", "test_operators.py"), "--accept"),
    )
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个命令行参数选项，如果设置，则将 "--skip-build" 或 "--skip_build" 置为 True
    parser.add_argument(
        "--skip-build",
        "--skip_build",
        action="store_true",
        help="Skip building pytorch",
    )
    # 解析命令行参数，并将解析结果传递给主函数 main
    main(parser.parse_args())
```