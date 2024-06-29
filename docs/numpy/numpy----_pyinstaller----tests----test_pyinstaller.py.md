# `.\numpy\numpy\_pyinstaller\tests\test_pyinstaller.py`

```
# 导入必要的模块
import subprocess  # 子进程管理模块，用于执行外部命令
from pathlib import Path  # 提供处理文件和目录路径的类和函数

import pytest  # 测试框架 pytest


# 忽略 PyInstaller 中关于 'imp' 替换为 'importlib' 的警告
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
# 忽略关于 io.BytesIO() 泄漏的资源警告
@pytest.mark.filterwarnings('ignore::ResourceWarning')
# 参数化测试，分别测试 '--onedir' 和 '--onefile' 两种模式
@pytest.mark.parametrize("mode", ["--onedir", "--onefile"])
@pytest.mark.slow  # 标记为慢速测试，可能需要较长时间运行
def test_pyinstaller(mode, tmp_path):
    """Compile and run pyinstaller-smoke.py using PyInstaller."""

    # 导入并运行 PyInstaller 主程序
    pyinstaller_cli = pytest.importorskip("PyInstaller.__main__").run

    # 获取要编译的源文件的路径，并确保其为绝对路径
    source = Path(__file__).with_name("pyinstaller-smoke.py").resolve()

    # 构造 PyInstaller 的命令行参数列表
    args = [
        # 将所有生成的文件放置在 tmp_path 指定的路径中
        '--workpath', str(tmp_path / "build"),
        '--distpath', str(tmp_path / "dist"),
        '--specpath', str(tmp_path),
        mode,  # 使用当前参数化的模式
        str(source),  # 源文件的绝对路径
    ]

    # 调用 PyInstaller 主程序并传入参数执行编译
    pyinstaller_cli(args)

    # 根据模式选择生成的可执行文件路径
    if mode == "--onefile":
        exe = tmp_path / "dist" / source.stem
    else:
        exe = tmp_path / "dist" / source.stem / source.stem

    # 运行生成的可执行文件，并获取其输出结果
    p = subprocess.run([str(exe)], check=True, stdout=subprocess.PIPE)
    # 断言输出结果是否为预期的 "I made it!"
    assert p.stdout.strip() == b"I made it!"
```