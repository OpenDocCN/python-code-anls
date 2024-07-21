# `.\pytorch\test\torch_np\check_tests_conform.py`

```py
import sys
import textwrap
from pathlib import Path


def check(path):
    """Check a test file for common issues with pytest->pytorch conversion."""
    # 打印文件名
    print(path.name)
    # 打印与文件名同样长度的分隔线
    print("=" * len(path.name), "\n")

    # 读取文件内容，并按行拆分为列表
    src = path.read_text().split("\n")
    for num, line in enumerate(src):
        # 如果是注释行，则跳过
        if is_comment(line):
            continue

        # 模块级别的测试函数
        if line.startswith("def test"):
            report_violation(line, num, header="Module-level test function")

        # 测试类必须继承自 TestCase
        if line.startswith("class Test") and "TestCase" not in line:
            report_violation(
                line, num, header="Test class does not inherit from TestCase"
            )

        # pytest 特定的一些内容
        if "pytest.mark" in line:
            report_violation(line, num, header="pytest.mark.something")

        # 检查是否包含 pytest 相关的标记，例如 pytest.xfail, pytest.skip, pytest.param
        for part in ["pytest.xfail", "pytest.skip", "pytest.param"]:
            if part in line:
                report_violation(line, num, header=f"stray {part}")

        # 检查是否以 @parametrize 开始，表示参数化测试
        if textwrap.dedent(line).startswith("@parametrize"):
            # 回溯检查
            nn = num
            for nn in range(num, -1, -1):
                ln = src[nn]
                if "class Test" in ln:
                    # hack: 大缩进 => 可能是内部类
                    if len(ln) - len(ln.lstrip()) < 8:
                        break
            else:
                report_violation(line, num, "off-class parametrize")
            if not src[nn - 1].startswith("@instantiate_parametrized_tests"):
                report_violation(
                    line, num, f"missing instantiation of parametrized tests in {ln}?"
                )


def is_comment(line):
    # 检查是否为注释行
    return textwrap.dedent(line).startswith("#")


def report_violation(line, lineno, header):
    # 打印违规信息
    print(f">>>> line {lineno} : {header}\n {line}\n")


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        raise ValueError("Usage : python check_tests_conform path/to/file/or/dir")

    path = Path(argv[1])

    if path.is_dir():
        # 对目录下的所有文件运行检查（不包括子目录）
        for this_path in path.glob("test*.py"):
            check(this_path)
    else:
        check(path)
```