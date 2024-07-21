# `.\pytorch\tools\extract_scripts.py`

```
#!/usr/bin/env python3
# 声明脚本解释器为 Python 3，并使脚本可执行

from __future__ import annotations
# 导入未来版本的类型注解支持

import argparse
# 导入用于解析命令行参数的模块
import re
# 导入用于正则表达式操作的模块
import sys
# 导入与系统交互相关的模块
from pathlib import Path
# 导入用于处理路径的模块
from typing import Any, Dict
# 导入类型提示相关的模块
from typing_extensions import TypedDict  # Python 3.11+
# 导入用于类型字典支持的模块

import yaml
# 导入用于 YAML 文件操作的模块

Step = Dict[str, Any]
# 定义 Step 类型别名，表示步骤字典的结构

class Script(TypedDict):
    extension: str
    script: str
# 定义 Script 类型字典，包含 extension 和 script 两个字段

def extract(step: Step) -> Script | None:
    run = step.get("run")
    # 获取步骤中的 "run" 命令

    # https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#using-a-specific-shell
    shell = step.get("shell", "bash")
    # 获取步骤中的 shell 选项，默认为 "bash"
    extension = {
        "bash": ".sh",
        "pwsh": ".ps1",
        "python": ".py",
        "sh": ".sh",
        "cmd": ".cmd",
        "powershell": ".ps1",
    }.get(shell)
    # 根据 shell 类型选择相应的文件扩展名

    is_gh_script = step.get("uses", "").startswith("actions/github-script@")
    # 检查是否使用了 GitHub 脚本动作

    gh_script = step.get("with", {}).get("script")
    # 获取 GitHub 脚本的具体内容

    if run is not None and extension is not None:
        script = {
            "bash": f"#!/usr/bin/env bash\nset -eo pipefail\n{run}",
            "sh": f"#!/usr/bin/env sh\nset -e\n{run}",
        }.get(shell, run)
        # 根据 shell 类型生成脚本内容

        return {"extension": extension, "script": script}
    elif is_gh_script and gh_script is not None:
        return {"extension": ".js", "script": gh_script}
        # 如果是 GitHub 脚本动作，则返回 JavaScript 扩展名和脚本内容
    else:
        return None
        # 如果无法提取有效脚本，则返回 None

def main() -> None:
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象
    parser.add_argument("--out", required=True)
    # 添加必需的 --out 参数选项
    args = parser.parse_args()
    # 解析命令行参数

    out = Path(args.out)
    # 将参数转换为 Path 对象
    if out.exists():
        sys.exit(f"{out} already exists; aborting to avoid overwriting")
        # 如果输出路径已存在，则退出程序

    gha_expressions_found = False
    # 初始化 GitHub Actions 表达式是否发现的标志

    for p in Path(".github/workflows").iterdir():
        # 遍历 .github/workflows 目录下的文件
        with open(p, "rb") as f:
            workflow = yaml.safe_load(f)
            # 加载 YAML 文件内容

        for job_name, job in workflow["jobs"].items():
            # 遍历工作流中的每个作业
            job_dir = out / p / job_name
            # 创建作业的目录路径

            if "steps" not in job:
                continue
                # 如果作业中没有步骤，继续下一个作业

            steps = job["steps"]
            # 获取作业的步骤列表
            index_chars = len(str(len(steps) - 1))
            # 计算步骤数的字符长度，用于格式化文件名

            for i, step in enumerate(steps, start=1):
                # 遍历作业中的每个步骤
                extracted = extract(step)
                # 提取步骤中的脚本内容和扩展名

                if extracted:
                    script = extracted["script"]
                    # 获取提取的脚本内容
                    step_name = step.get("name", "")
                    # 获取步骤的名称，如果没有则为空字符串

                    if "${{" in script:
                        gha_expressions_found = True
                        # 如果脚本中包含 GitHub Actions 表达式，则设置标志并打印警告信息

                        print(
                            f"{p} job `{job_name}` step {i}: {step_name}",
                            file=sys.stderr,
                        )
                        # 打印包含 GitHub Actions 表达式的警告信息到标准错误流

                    job_dir.mkdir(parents=True, exist_ok=True)
                    # 创建作业目录，如果不存在则递归创建

                    sanitized = re.sub(
                        "[^a-zA-Z_]+",
                        "_",
                        f"_{step_name}",
                    ).rstrip("_")
                    # 使用正则表达式清理步骤名称，生成安全的文件名前缀

                    extension = extracted["extension"]
                    # 获取脚本文件的扩展名
                    filename = f"{i:0{index_chars}}{sanitized}{extension}"
                    # 根据格式化索引和清理后的步骤名称生成文件名

                    (job_dir / filename).write_text(script)
                    # 将脚本内容写入文件
    # 如果发现了 GitHub Actions 表达式
    if gha_expressions_found:
        # 输出错误信息并退出程序，提示用户替换 GitHub Actions 表达式为 `env` 变量，以确保安全性
        sys.exit(
            "Each of the above scripts contains a GitHub Actions "
            "${{ <expression> }} which must be replaced with an `env` variable"
            " for security reasons."
        )
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```