# `.\numpy\numpy\_core\tests\test_limited_api.py`

```py
# 导入标准库和第三方库
import os                      # 导入操作系统功能模块
import shutil                  # 导入文件和目录操作模块
import subprocess              # 导入子进程管理模块
import sys                     # 导入系统相关功能模块
import sysconfig               # 导入 Python 配置信息模块
import pytest                  # 导入 pytest 测试框架

from numpy.testing import IS_WASM, IS_PYPY, NOGIL_BUILD, IS_EDITABLE  # 导入 numpy 测试相关标志

# 以下代码段是从 random.tests.test_extending 复制过来的
try:
    import cython             # 尝试导入 Cython 模块
    from Cython.Compiler.Version import version as cython_version  # 导入 Cython 版本信息
except ImportError:
    cython = None             # 如果导入失败，将 cython 设为 None
else:
    from numpy._utils import _pep440  # 导入 numpy 内部工具函数

    # 注意：需要与 pyproject.toml 中的版本保持同步
    required_version = "3.0.6"
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # 如果 Cython 版本过低或者不符合要求，则跳过该测试
        cython = None

# 使用 pytest.mark.skipif 标记，如果 cython 为 None，则跳过测试，给出原因为需要 Cython
pytestmark = pytest.mark.skipif(cython is None, reason="requires cython")

# 如果是可编辑安装（editable install），则跳过测试并给出原因，这种安装方式不支持需要编译步骤的测试
if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )

# 使用 pytest.fixture(scope='module') 标记一个模块级别的 fixture
def install_temp(tmpdir_factory):
    # 基于 random.tests.test_extending 中的 test_cython 的部分内容
    if IS_WASM:
        pytest.skip("No subprocess")  # 如果是在 WASM 环境下，则跳过测试

    # 定义源代码目录和构建目录
    srcdir = os.path.join(os.path.dirname(__file__), 'examples', 'limited_api')
    build_dir = tmpdir_factory.mktemp("limited_api") / "build"
    os.makedirs(build_dir, exist_ok=True)  # 创建构建目录，如果不存在则创建

    try:
        subprocess.check_call(["meson", "--version"])  # 检查 meson 的版本
    except FileNotFoundError:
        pytest.skip("No usable 'meson' found")  # 如果找不到可用的 meson，则跳过测试

    # 根据操作系统不同，执行不同的 meson 构建命令
    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release",
                               "--vsenv", str(srcdir)],
                              cwd=build_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup", str(srcdir)],
                              cwd=build_dir
                              )

    try:
        subprocess.check_call(["meson", "compile", "-vv"], cwd=build_dir)  # 编译项目
    except subprocess.CalledProcessError as p:
        print(f"{p.stdout=}")  # 打印出错时的标准输出
        print(f"{p.stderr=}")  # 打印出错时的标准错误
        raise  # 抛出异常

    sys.path.append(str(build_dir))  # 将构建目录添加到系统路径中，以便导入测试所需的模块

# 使用 pytest.mark.skipif 标记，如果是在 WASM 环境下，则跳过测试
@pytest.mark.skipif(IS_WASM, reason="Can't start subprocess")
# 使用 pytest.mark.xfail 标记，如果 Py_DEBUG 开启，则预期测试失败，给出原因
@pytest.mark.xfail(
    sysconfig.get_config_var("Py_DEBUG"),
    reason=(
        "Py_LIMITED_API is incompatible with Py_DEBUG, Py_TRACE_REFS, "
        "and Py_REF_DEBUG"
    ),
)
# 使用 pytest.mark.xfail 标记，如果是 NOGIL_BUILD，则预期测试失败，给出原因
@pytest.mark.xfail(
    NOGIL_BUILD,
    reason="Py_GIL_DISABLED builds do not currently support the limited API",
)
# 使用 pytest.mark.skipif 标记，如果是在 PyPy 环境下，则跳过测试
@pytest.mark.skipif(IS_PYPY, reason="no support for limited API in PyPy")
# 定义测试函数 test_limited_api，并使用 install_temp 作为 fixture
def test_limited_api(install_temp):
    """Test building a third-party C extension with the limited API
    and building a cython extension with the limited API
    """

    import limited_api1  # 导入测试用的 limited_api1 模块
    import limited_api2  # 导入测试用的 limited_api2 模块
```