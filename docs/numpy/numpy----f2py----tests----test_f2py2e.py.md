# `.\numpy\numpy\f2py\tests\test_f2py2e.py`

```py
# 导入必要的模块和库
import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform

# 导入 pytest 测试框架
import pytest

# 导入本地的 util 模块
from . import util

# 导入 numpy 中的 f2py2e 模块的主函数
from numpy.f2py.f2py2e import main as f2pycli

#########################
# CLI utils and classes #
#########################

# 定义一个命名元组 PPaths，表示路径集合
PPaths = namedtuple("PPaths", "finp, f90inp, pyf, wrap77, wrap90, cmodf")


def get_io_paths(fname_inp, mname="untitled"):
    """获取输入文件的路径和生成文件的可能路径

    这个函数用于生成测试时需要的输入和输出文件路径。

    ..note::

         由于这并不实际运行 f2py，因此生成的文件并不一定存在，模块名通常也是错误的

    Parameters
    ----------
    fname_inp : str
                输入文件名
    mname : str, optional
                模块名，默认为 untitled

    Returns
    -------
    genp : NamedTuple PPaths
            可能的生成文件路径集合，不一定全部存在
    """
    # 将输入文件名转换为 Path 对象
    bpath = Path(fname_inp)
    # 返回可能的生成文件路径的命名元组
    return PPaths(
        finp=bpath.with_suffix(".f"),
        f90inp=bpath.with_suffix(".f90"),
        pyf=bpath.with_suffix(".pyf"),
        wrap77=bpath.with_name(f"{mname}-f2pywrappers.f"),
        wrap90=bpath.with_name(f"{mname}-f2pywrappers2.f90"),
        cmodf=bpath.with_name(f"{mname}module.c"),
    )


##############
# CLI Fixtures and Tests #
##############

# 定义 pytest 的 session 级别的 fixture：生成一个 hello_world_f90 的测试文件
@pytest.fixture(scope="session")
def hello_world_f90(tmpdir_factory):
    """生成一个用于测试的单个 f90 文件"""
    # 从 util 模块中读取文件内容
    fdat = util.getpath("tests", "src", "cli", "hiworld.f90").read_text()
    # 创建临时文件路径
    fn = tmpdir_factory.getbasetemp() / "hello.f90"
    # 将文件内容写入临时文件
    fn.write_text(fdat, encoding="ascii")
    # 返回临时文件路径
    return fn


# 定义 pytest 的 session 级别的 fixture：生成一个 gh23598_warn 的测试文件
@pytest.fixture(scope="session")
def gh23598_warn(tmpdir_factory):
    """用于测试 gh23598 中警告的 f90 文件"""
    # 从 util 模块中读取文件内容
    fdat = util.getpath("tests", "src", "crackfortran", "gh23598Warn.f90").read_text()
    # 创建临时文件路径
    fn = tmpdir_factory.getbasetemp() / "gh23598Warn.f90"
    # 将文件内容写入临时文件
    fn.write_text(fdat, encoding="ascii")
    # 返回临时文件路径
    return fn


# 定义 pytest 的 session 级别的 fixture：生成一个 gh22819_cli 的测试文件
@pytest.fixture(scope="session")
def gh22819_cli(tmpdir_factory):
    """用于测试 ghff819 中不允许的 CLI 参数的 f90 文件"""
    # 从 util 模块中读取文件内容
    fdat = util.getpath("tests", "src", "cli", "gh_22819.pyf").read_text()
    # 创建临时文件路径
    fn = tmpdir_factory.getbasetemp() / "gh_22819.pyf"
    # 将文件内容写入临时文件
    fn.write_text(fdat, encoding="ascii")
    # 返回临时文件路径
    return fn


# 定义 pytest 的 session 级别的 fixture：生成一个 hello_world_f77 的测试文件
@pytest.fixture(scope="session")
def hello_world_f77(tmpdir_factory):
    """生成一个用于测试的单个 f77 文件"""
    # 从 util 模块中读取文件内容
    fdat = util.getpath("tests", "src", "cli", "hi77.f").read_text()
    # 创建临时文件路径
    fn = tmpdir_factory.getbasetemp() / "hello.f"
    # 将文件内容写入临时文件
    fn.write_text(fdat, encoding="ascii")
    # 返回临时文件路径
    return fn


# 定义 pytest 的 session 级别的 fixture：生成一个 retreal_f77 的测试文件
@pytest.fixture(scope="session")
def retreal_f77(tmpdir_factory):
    """生成一个用于测试的单个 f77 文件"""
    # 从 util 模块中读取文件内容
    fdat = util.getpath("tests", "src", "return_real", "foo77.f").read_text()
    # 创建临时文件路径
    fn = tmpdir_factory.getbasetemp() / "foo.f"
    # 将文件内容写入临时文件
    fn.write_text(fdat, encoding="ascii")
    # 返回临时文件路径
    return fn
# 定义一个 pytest 的 session 级别的 fixture，用于生成单个的 f90 文件用于测试
@pytest.fixture(scope="session")
def f2cmap_f90(tmpdir_factory):
    # 从指定路径读取 f90 文件的内容作为字符串
    fdat = util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90").read_text()
    # 读取另一个路径下的文件内容作为字符串
    f2cmap = util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap").read_text()
    # 获取临时目录并创建一个新的 f90 文件，写入之前读取的 fdat 内容
    fn = tmpdir_factory.getbasetemp() / "f2cmap.f90"
    # 创建一个名为 mapfile 的文件，并将 f2cmap 的内容写入其中
    fmap = tmpdir_factory.getbasetemp() / "mapfile"
    fn.write_text(fdat, encoding="ascii")
    fmap.write_text(f2cmap, encoding="ascii")
    # 返回生成的 f90 文件的路径
    return fn


# 测试函数，检查模块名称是否正确处理
def test_gh22819_cli(capfd, gh22819_cli, monkeypatch):
    """Check that module names are handled correctly
    gh-22819
    Essentially, the -m name cannot be used to import the module, so the module
    named in the .pyf needs to be used instead

    CLI :: -m and a .pyf file
    """
    # 获取路径对象
    ipath = Path(gh22819_cli)
    # 设置模拟的命令行参数
    monkeypatch.setattr(sys, "argv", f"f2py -m blah {ipath}".split())
    # 在指定路径下执行 f2pycli 命令，并验证生成的文件是否符合预期
    with util.switchdir(ipath.parent):
        f2pycli()
        # 获取生成文件的名称列表
        gen_paths = [item.name for item in ipath.parent.rglob("*") if item.is_file()]
        # 断言生成的特定文件不存在
        assert "blahmodule.c" not in gen_paths
        assert "blah-f2pywrappers.f" not in gen_paths
        # 断言生成的特定文件存在
        assert "test_22819-f2pywrappers.f" in gen_paths
        assert "test_22819module.c" in gen_paths
        assert "Ignoring blah"


# 测试函数，检查是否只允许一个 .pyf 文件存在
def test_gh22819_many_pyf(capfd, gh22819_cli, monkeypatch):
    """Only one .pyf file allowed
    gh-22819
    CLI :: .pyf files
    """
    # 获取路径对象
    ipath = Path(gh22819_cli)
    # 设置模拟的命令行参数
    monkeypatch.setattr(sys, "argv", f"f2py -m blah {ipath} hello.pyf".split())
    # 在指定路径下执行 f2pycli 命令，预期会引发 ValueError 异常
    with util.switchdir(ipath.parent):
        with pytest.raises(ValueError, match="Only one .pyf file per call"):
            f2pycli()


# 测试函数，检查是否生成警告信息
def test_gh23598_warn(capfd, gh23598_warn, monkeypatch):
    # 获取输出路径对象
    foutl = get_io_paths(gh23598_warn, mname="test")
    ipath = foutl.f90inp
    # 设置模拟的命令行参数
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test'.split())

    with util.switchdir(ipath.parent):
        f2pycli()  # 生成文件
        # 读取生成的包装文件内容
        wrapper = foutl.wrap90.read_text()
        # 断言特定字符串不在包装文件中
        assert "intproductf2pywrap, intpr" not in wrapper


# 测试函数，检查是否生成签名文件
def test_gen_pyf(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file is generated via the CLI
    CLI :: -h
    """
    # 获取路径对象
    ipath = Path(hello_world_f90)
    # 定义签名文件的输出路径
    opath = Path(hello_world_f90).stem + ".pyf"
    # 设置模拟的命令行参数
    monkeypatch.setattr(sys, "argv", f'f2py -h {opath} {ipath}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()  # 生成包装文件
        out, _ = capfd.readouterr()
        # 断言输出包含特定提示信息
        assert "Saving signatures to file" in out
        # 断言签名文件是否存在
        assert Path(f'{opath}').exists()


# 测试函数，检查是否可以将签名文件输出到 stdout
def test_gen_pyf_stdout(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file can be dumped to stdout
    CLI :: -h
    """
    # 获取路径对象
    ipath = Path(hello_world_f90)
    # 设置模拟的命令行参数
    monkeypatch.setattr(sys, "argv", f'f2py -h stdout {ipath}'.split())
    # 使用 `util.switchdir(ipath.parent)` 进入 `ipath` 的父目录环境，执行以下代码块
    with util.switchdir(ipath.parent):
        # 调用 `f2pycli()` 函数，通常用于与Fortran程序交互的命令行接口
        f2pycli()
        # 读取并捕获标准输出和错误输出
        out, _ = capfd.readouterr()
        # 断言标准输出中包含特定文本 "Saving signatures to file"
        assert "Saving signatures to file" in out
        # 断言标准输出中包含特定文本 "function hi() ! in "
        assert "function hi() ! in " in out
# 确保CLI在不覆盖签名文件的情况下拒绝操作
def test_gen_pyf_no_overwrite(capfd, hello_world_f90, monkeypatch):
    """Ensures that the CLI refuses to overwrite signature files
    CLI :: -h without --overwrite-signature
    """
    # 获取指定Fortran文件的路径对象
    ipath = Path(hello_world_f90)
    # 通过monkeypatch设置模拟的命令行参数
    monkeypatch.setattr(sys, "argv", f'f2py -h faker.pyf {ipath}'.split())

    # 在指定路径下进行上下文切换
    with util.switchdir(ipath.parent):
        # 创建名为"faker.pyf"的文件，内容为"Fake news"，ASCII编码
        Path("faker.pyf").write_text("Fake news", encoding="ascii")
        # 确保引发SystemExit异常
        with pytest.raises(SystemExit):
            # 调用f2pycli()函数，预期拒绝覆盖操作
            f2pycli()
            # 读取并捕获标准输出和标准错误
            _, err = capfd.readouterr()
            # 断言标准错误中包含特定提示信息
            assert "Use --overwrite-signature to overwrite" in err


@pytest.mark.skipif((platform.system() != 'Linux') or (sys.version_info <= (3, 12)),
                    reason='Compiler and 3.12 required')
def test_untitled_cli(capfd, hello_world_f90, monkeypatch):
    """Check that modules are named correctly

    CLI :: defaults
    """
    # 获取指定Fortran文件的路径对象
    ipath = Path(hello_world_f90)
    # 通过monkeypatch设置模拟的命令行参数
    monkeypatch.setattr(sys, "argv", f"f2py --backend meson -c {ipath}".split())
    # 在指定路径下进行上下文切换
    with util.switchdir(ipath.parent):
        # 调用f2pycli()函数，检查默认情况下模块命名是否正确
        f2pycli()
        # 读取并捕获标准输出和标准错误
        out, _ = capfd.readouterr()
        # 断言标准输出中包含"untitledmodule.c"
        assert "untitledmodule.c" in out


@pytest.mark.skipif((platform.system() != 'Linux') or (sys.version_info <= (3, 12)), reason='Compiler and 3.12 required')
def test_no_py312_distutils_fcompiler(capfd, hello_world_f90, monkeypatch):
    """Check that no distutils imports are performed on 3.12
    CLI :: --fcompiler --help-link --backend distutils
    """
    # 设置模块名
    MNAME = "hi"
    # 获取输入和输出文件路径对象
    foutl = get_io_paths(hello_world_f90, mname=MNAME)
    ipath = foutl.f90inp
    # 通过monkeypatch设置模拟的命令行参数
    monkeypatch.setattr(
        sys, "argv", f"f2py {ipath} -c --fcompiler=gfortran -m {MNAME}".split()
    )
    # 在指定路径下进行上下文切换
    with util.switchdir(ipath.parent):
        # 调用f2pycli()函数，检查是否使用了不兼容meson的--fcompiler选项
        f2pycli()
        # 读取并捕获标准输出和标准错误
        out, _ = capfd.readouterr()
        # 断言标准输出中包含特定提示信息
        assert "--fcompiler cannot be used with meson" in out
    # 通过monkeypatch设置模拟的命令行参数
    monkeypatch.setattr(
        sys, "argv", f"f2py --help-link".split()
    )
    # 在指定路径下进行上下文切换
    with util.switchdir(ipath.parent):
        # 调用f2pycli()函数，检查是否正确处理了--help-link选项
        f2pycli()
        # 读取并捕获标准输出和标准错误
        out, _ = capfd.readouterr()
        # 断言标准输出中包含特定提示信息
        assert "Use --dep for meson builds" in out
    # 设置新的模块名
    MNAME = "hi2" # 需要一个不同的模块名来进行新的-c操作
    # 通过monkeypatch设置模拟的命令行参数
    monkeypatch.setattr(
        sys, "argv", f"f2py {ipath} -c -m {MNAME} --backend distutils".split()
    )
    # 在指定路径下进行上下文切换
    with util.switchdir(ipath.parent):
        # 调用f2pycli()函数，检查是否正确处理了不兼容Python>=3.12的--backend distutils选项
        f2pycli()
        # 读取并捕获标准输出和标准错误
        out, _ = capfd.readouterr()
        # 断言标准输出中包含特定提示信息
        assert "Cannot use distutils backend with Python>=3.12" in out


@pytest.mark.xfail
def test_f2py_skip(capfd, retreal_f77, monkeypatch):
    """Tests that functions can be skipped
    CLI :: skip:
    """
    # 获取输入文件路径对象和要跳过的函数名称
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    remaining = "td s0"
    # 通过monkeypatch设置模拟的命令行参数
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test skip: {toskip}'.split())
    # 在指定路径中切换工作目录并执行 f2pycli 命令
    with util.switchdir(ipath.parent):
        # 调用名为 f2pycli 的函数，执行相关操作
        f2pycli()
        # 从捕获的标准输出和标准错误中读取输出和错误信息
        out, err = capfd.readouterr()
        # 遍历需要跳过的任务列表，检查错误信息中是否包含特定字符串
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not found the body of interfaced routine "{skey}". Skipping.'
                in err)
        # 遍历剩余任务列表，检查输出信息中是否包含特定字符串
        for rkey in remaining.split():
            assert f'Constructing wrapper function "{rkey}"' in out
def test_f2py_only(capfd, retreal_f77, monkeypatch):
    """Test that functions can be kept by only:
    CLI :: only:
    """
    # 获取测试所需的输入输出路径对象
    foutl = get_io_paths(retreal_f77, mname="test")
    # 获取输入路径
    ipath = foutl.finp
    # 需要跳过的例外例程
    toskip = "t0 t4 t8 sd s8 s4"
    # 需要保留的例程
    tokeep = "td s0"
    # 使用 monkeypatch 设置模拟的命令行参数
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test only: {tokeep}'.split())

    # 在指定的目录中执行函数 f2pycli
    with util.switchdir(ipath.parent):
        f2pycli()
        # 读取标准输出和标准错误
        out, err = capfd.readouterr()
        # 检查需要跳过的例外例程是否在错误信息中
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not find the body of interfaced routine "{skey}". Skipping.'
                in err)
        # 检查需要保留的例程是否在输出信息中
        for rkey in tokeep.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_file_processing_switch(capfd, hello_world_f90, retreal_f77,
                                monkeypatch):
    """Tests that it is possible to return to file processing mode
    CLI :: :
    BUG: numpy-gh #20520
    """
    # 获取测试所需的输入输出路径对象
    foutl = get_io_paths(retreal_f77, mname="test")
    # 获取输入路径
    ipath = foutl.finp
    # 需要跳过的例外例程
    toskip = "t0 t4 t8 sd s8 s4"
    # 获取另一个输入路径
    ipath2 = Path(hello_world_f90)
    # 需要保留的例程，以及其中的一个例程在 ipath2 中
    tokeep = "td s0 hi"  # hi is in ipath2
    # 模块名
    mname = "blah"
    # 使用 monkeypatch 设置模拟的命令行参数
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -m {mname} only: {tokeep} : {ipath2}'.split(),
    )

    # 在指定的目录中执行函数 f2pycli
    with util.switchdir(ipath.parent):
        f2pycli()
        # 读取标准输出和标准错误
        out, err = capfd.readouterr()
        # 检查需要跳过的例外例程是否在错误信息中
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not find the body of interfaced routine "{skey}". Skipping.'
                in err)
        # 检查需要保留的例程是否在输出信息中
        for rkey in tokeep.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_mod_gen_f77(capfd, hello_world_f90, monkeypatch):
    """Checks the generation of files based on a module name
    CLI :: -m
    """
    # 模块名
    MNAME = "hi"
    # 获取测试所需的输入输出路径对象
    foutl = get_io_paths(hello_world_f90, mname=MNAME)
    # 获取输入路径
    ipath = foutl.f90inp
    # 使用 monkeypatch 设置模拟的命令行参数
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m {MNAME}'.split())
    # 在指定的目录中执行函数 f2pycli
    with util.switchdir(ipath.parent):
        f2pycli()

    # 检查是否生成了 C 模块文件
    assert Path.exists(foutl.cmodf)
    # 检查文件是否包含函数，以便检查 F77 封装器
    assert Path.exists(foutl.wrap77)


def test_mod_gen_gh25263(capfd, hello_world_f77, monkeypatch):
    """Check that pyf files are correctly generated with module structure
    CLI :: -m <name> -h pyf_file
    BUG: numpy-gh #20520
    """
    # 模块名
    MNAME = "hi"
    # 获取测试所需的输入输出路径对象
    foutl = get_io_paths(hello_world_f77, mname=MNAME)
    # 获取输入路径
    ipath = foutl.finp
    # 使用 monkeypatch 设置模拟的命令行参数
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m {MNAME} -h hi.pyf'.split())
    # 在指定的目录中执行函数 f2pycli
    with util.switchdir(ipath.parent):
        f2pycli()
        # 打开生成的 hi.pyf 文件，读取内容进行检查
        with Path('hi.pyf').open() as hipyf:
            pyfdat = hipyf.read()
            # 断言生成的 pyf 文件包含指定的模块名信息
            assert "python module hi" in pyfdat


def test_lower_cmod(capfd, hello_world_f77, monkeypatch):
    """Lowers cases by flag or when -h is present

    CLI :: --[no-]lower
    """
    # 获取测试所需的输入输出路径对象
    foutl = get_io_paths(hello_world_f77, mname="test")
    # 获取输入路径
    ipath = foutl.finp
    # 编译正则表达式，匹配形如 "HI()" 的字符串
    capshi = re.compile(r"HI\(\)")
    # 编译正则表达式，匹配形如 "hi()" 的字符串
    capslo = re.compile(r"hi\(\)")
    
    # Case I: 当传递了 --lower 参数
    # 使用 monkeypatch 修改 sys.argv，模拟命令行参数为 'f2py <ipath> -m test --lower'
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m test --lower'.split())
    # 在 ipath.parent 目录下执行代码块
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 读取并捕获输出
        out, _ = capfd.readouterr()
        # 断言输出中应存在形如 "hi()" 的字符串
        assert capslo.search(out) is not None
        # 断言输出中不应存在形如 "HI()" 的字符串
        assert capshi.search(out) is None
    
    # Case II: 当传递了 --no-lower 参数
    # 使用 monkeypatch 修改 sys.argv，模拟命令行参数为 'f2py <ipath> -m test --no-lower'
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m test --no-lower'.split())
    # 在 ipath.parent 目录下执行代码块
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 读取并捕获输出
        out, _ = capfd.readouterr()
        # 断言输出中不应存在形如 "hi()" 的字符串
        assert capslo.search(out) is None
        # 断言输出中应存在形如 "HI()" 的字符串
        assert capshi.search(out) is not None
def test_lower_sig(capfd, hello_world_f77, monkeypatch):
    """Lowers cases in signature files by flag or when -h is present

    CLI :: --[no-]lower -h
    """
    # 获取输入输出路径对象
    foutl = get_io_paths(hello_world_f77, mname="test")
    # 获取输入路径
    ipath = foutl.finp

    # Signature files
    # 定义正则表达式，用于匹配特定字符串
    capshi = re.compile(r"Block: HI")
    capslo = re.compile(r"Block: hi")

    # Case I: --lower is implied by -h
    # TODO: Clean up to prevent passing --overwrite-signature
    # 修改系统参数，模拟命令行输入
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature'.split(),
    )

    # 在指定目录下执行命令行工具
    with util.switchdir(ipath.parent):
        f2pycli()  # 调用f2pycli函数
        out, _ = capfd.readouterr()  # 读取标准输出和标准错误
        assert capslo.search(out) is not None  # 断言输出中存在"Block: hi"
        assert capshi.search(out) is None  # 断言输出中不存在"Block: HI"

    # Case II: --no-lower overrides -h
    # 修改系统参数，模拟命令行输入
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature --no-lower'
        .split(),
    )

    # 在指定目录下执行命令行工具
    with util.switchdir(ipath.parent):
        f2pycli()  # 调用f2pycli函数
        out, _ = capfd.readouterr()  # 读取标准输出和标准错误
        assert capslo.search(out) is None  # 断言输出中不存在"Block: hi"
        assert capshi.search(out) is not None  # 断言输出中存在"Block: HI"


def test_build_dir(capfd, hello_world_f90, monkeypatch):
    """Ensures that the build directory can be specified

    CLI :: --build-dir
    """
    # 获取输入路径对象
    ipath = Path(hello_world_f90)
    mname = "blah"
    odir = "tttmp"

    # 修改系统参数，模拟命令行输入
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --build-dir {odir}'.split())

    # 在指定目录下执行命令行工具
    with util.switchdir(ipath.parent):
        f2pycli()  # 调用f2pycli函数
        out, _ = capfd.readouterr()  # 读取标准输出和标准错误
        assert f"Wrote C/API module \"{mname}\"" in out  # 断言输出中包含特定信息


def test_overwrite(capfd, hello_world_f90, monkeypatch):
    """Ensures that the build directory can be specified

    CLI :: --overwrite-signature
    """
    # 获取输入路径对象
    ipath = Path(hello_world_f90)

    # 修改系统参数，模拟命令行输入
    monkeypatch.setattr(
        sys, "argv",
        f'f2py -h faker.pyf {ipath} --overwrite-signature'.split())

    # 在指定目录下创建文件并写入内容
    with util.switchdir(ipath.parent):
        Path("faker.pyf").write_text("Fake news", encoding="ascii")
        f2pycli()  # 调用f2pycli函数
        out, _ = capfd.readouterr()  # 读取标准输出和标准错误
        assert "Saving signatures to file" in out  # 断言输出中包含特定信息


def test_latexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --latex-doc
    """
    # 获取输入路径对象
    ipath = Path(hello_world_f90)
    mname = "blah"

    # 修改系统参数，模拟命令行输入
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --latex-doc'.split())

    # 在指定目录下执行命令行工具
    with util.switchdir(ipath.parent):
        f2pycli()  # 调用f2pycli函数
        out, _ = capfd.readouterr()  # 读取标准输出和标准错误
        assert "Documentation is saved to file" in out  # 断言输出中包含特定信息
        with Path(f"{mname}module.tex").open() as otex:
            assert "\\documentclass" in otex.read()  # 断言生成的TeX文件中包含特定内容


def test_nolatexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --no-latex-doc
    """
    # 获取输入路径对象
    ipath = Path(hello_world_f90)
    mname = "blah"
    # 使用 monkeypatch 模块设置 sys.argv，模拟命令行参数以调用 f2py 工具
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-latex-doc'.split())
    
    # 在指定的路径 ipath 的父目录下执行代码块
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数，执行 f2py 工具的命令
        f2pycli()
        # 读取并捕获标准输出和标准错误输出
        out, _ = capfd.readouterr()
        # 断言确保输出中不包含特定的文本 "Documentation is saved to file"
        assert "Documentation is saved to file" not in out
def test_shortlatex(capfd, hello_world_f90, monkeypatch):
    """Ensures that truncated documentation is written out

    TODO: Test to ensure this has no effect without --latex-doc
    CLI :: --latex-doc --short-latex
    """
    # 将文件路径转换为 Path 对象
    ipath = Path(hello_world_f90)
    # 定义模块名
    mname = "blah"
    # 使用 monkeypatch 设置 sys.argv 以模拟命令行参数
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py -m {mname} {ipath} --latex-doc --short-latex'.split(),
    )

    # 切换到指定路径的上级目录
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 读取 stdout 和 stderr
        out, _ = capfd.readouterr()
        # 断言输出中包含特定信息
        assert "Documentation is saved to file" in out
        # 打开生成的 LaTeX 文件，断言文件内容不包含 LaTeX 文档标识
        with Path(f"./{mname}module.tex").open() as otex:
            assert "\\documentclass" not in otex.read()


def test_restdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that RsT documentation is written out

    CLI :: --rest-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    # 使用 monkeypatch 设置 sys.argv 以模拟命令行参数
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --rest-doc'.split())

    # 切换到指定路径的上级目录
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 读取 stdout 和 stderr
        out, _ = capfd.readouterr()
        # 断言输出中包含特定信息
        assert "ReST Documentation is saved to file" in out
        # 打开生成的 ReST 文件，断言文件内容包含 ReST 标识
        with Path(f"./{mname}module.rest").open() as orst:
            assert r".. -*- rest -*-" in orst.read()


def test_norestexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --no-rest-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    # 使用 monkeypatch 设置 sys.argv 以模拟命令行参数
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-rest-doc'.split())

    # 切换到指定路径的上级目录
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 读取 stdout 和 stderr
        out, _ = capfd.readouterr()
        # 断言输出中不包含特定信息
        assert "ReST Documentation is saved to file" not in out


def test_debugcapi(capfd, hello_world_f90, monkeypatch):
    """Ensures that debugging wrappers are written

    CLI :: --debug-capi
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    # 使用 monkeypatch 设置 sys.argv 以模拟命令行参数
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --debug-capi'.split())

    # 切换到指定路径的上级目录
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 打开生成的 C 模块文件，断言文件内容包含调试相关标识
        with Path(f"./{mname}module.c").open() as ocmod:
            assert r"#define DEBUGCFUNCS" in ocmod.read()


@pytest.mark.skip(reason="Consistently fails on CI; noisy so skip not xfail.")
def test_debugcapi_bld(hello_world_f90, monkeypatch):
    """Ensures that debugging wrappers work

    CLI :: --debug-capi -c
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    # 使用 monkeypatch 设置 sys.argv 以模拟命令行参数
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} -c --debug-capi'.split())

    # 切换到指定路径的上级目录
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 构建 Python 命令来运行生成的模块，断言输出符合预期
        cmd_run = shlex.split("python3 -c \"import blah; blah.hi()\"")
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        eerr = textwrap.dedent("""\
debug-capi:Python C/API function blah.hi()
debug-capi:float hi=:output,hidden,scalar
debug-capi:hi=0
debug-capi:Fortran subroutine `f2pywraphi(&hi)'
# 设置一个 debug-capi 标签的日志条目，表明当前 hi 值为 0
debug-capi:hi=0
# 标记 debug-capi，指示正在构建返回值
debug-capi:Building return value.
# 记录 Python C/API 函数 blah.hi 的成功执行
debug-capi:Python C/API function blah.hi: successful.
# 释放内存的 debug-capi 标签日志条目
debug-capi:Freeing memory.
        """)
# 断言标准输出 rout.stdout 等于期望输出 eout
        assert rout.stdout == eout
# 断言标准错误 rout.stderr 等于期望错误输出 eerr
        assert rout.stderr == eerr


def test_wrapfunc_def(capfd, hello_world_f90, monkeypatch):
    """Ensures that fortran subroutine wrappers for F77 are included by default

    CLI :: --[no]-wrap-functions
    """
    # 隐式设置
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv", f'f2py -m {mname} {ipath}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
    out, _ = capfd.readouterr()
    assert r"Fortran 77 wrappers are saved to" in out

    # 显式设置
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --wrap-functions'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" in out


def test_nowrapfunc(capfd, hello_world_f90, monkeypatch):
    """Ensures that fortran subroutine wrappers for F77 can be disabled

    CLI :: --no-wrap-functions
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-wrap-functions'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" not in out


def test_inclheader(capfd, hello_world_f90, monkeypatch):
    """Add to the include directories

    CLI :: -include
    TODO: Document this in the help string
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py -m {mname} {ipath} -include<stdbool.h> -include<stdio.h> '.
        split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        with Path(f"./{mname}module.c").open() as ocmod:
            ocmr = ocmod.read()
            assert "#include <stdbool.h>" in ocmr
            assert "#include <stdio.h>" in ocmr


def test_inclpath():
    """Add to the include directories

    CLI :: --include-paths
    """
    # TODO: populate
    pass


def test_hlink():
    """Add to the include directories

    CLI :: --help-link
    """
    # TODO: populate
    pass


def test_f2cmap(capfd, f2cmap_f90, monkeypatch):
    """Check that Fortran-to-Python KIND specs can be passed

    CLI :: --f2cmap
    """
    ipath = Path(f2cmap_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --f2cmap mapfile'.split())
    # 切换工作目录到输入路径的父目录
    with util.switchdir(ipath.parent):
        # 调用名为 f2pycli 的函数
        f2pycli()
        # 读取并捕获标准输出和错误输出
        out, _ = capfd.readouterr()
        # 断言输出中包含特定的字符串
        assert "Reading f2cmap from 'mapfile' ..." in out
        assert "Mapping \"real(kind=real32)\" to \"float\"" in out
        assert "Mapping \"real(kind=real64)\" to \"double\"" in out
        assert "Mapping \"integer(kind=int64)\" to \"long_long\"" in out
        assert "Successfully applied user defined f2cmap changes" in out
# 函数定义：测试函数，用于验证是否成功减少输出的详细程度
def test_quiet(capfd, hello_world_f90, monkeypatch):
    """Reduce verbosity

    CLI :: --quiet
    """
    # 获取文件路径对象
    ipath = Path(hello_world_f90)
    # 设置模拟的命令行参数，将输出设置为静默模式
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --quiet'.split())

    # 在文件路径的父目录中执行测试
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 读取并捕获标准输出和标准错误
        out, _ = capfd.readouterr()
        # 断言标准输出为空
        assert len(out) == 0


# 函数定义：测试函数，用于验证是否成功增加输出的详细程度
def test_verbose(capfd, hello_world_f90, monkeypatch):
    """Increase verbosity

    CLI :: --verbose
    """
    # 获取文件路径对象
    ipath = Path(hello_world_f90)
    # 设置模拟的命令行参数，将输出设置为详细模式
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --verbose'.split())

    # 在文件路径的父目录中执行测试
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 读取并捕获标准输出和标准错误
        out, _ = capfd.readouterr()
        # 断言标准输出中包含指定字符串
        assert "analyzeline" in out


# 函数定义：测试函数，用于验证版本输出
def test_version(capfd, monkeypatch):
    """Ensure version

    CLI :: -v
    """
    # 设置模拟的命令行参数，获取版本信息
    monkeypatch.setattr(sys, "argv", 'f2py -v'.split())
    # 期望抛出 SystemExit 异常
    with pytest.raises(SystemExit):
        # 调用 f2pycli 函数
        f2pycli()
        # 读取并捕获标准输出和标准错误
        out, _ = capfd.readouterr()
        # 导入 NumPy 库，断言其版本号与输出的版本号一致
        import numpy as np
        assert np.__version__ == out.strip()


# 函数定义：跳过的测试函数，因为在持续集成中经常失败，不进行预期失败标记
@pytest.mark.skip(reason="Consistently fails on CI; noisy so skip not xfail.")
def test_npdistop(hello_world_f90, monkeypatch):
    """
    CLI :: -c
    """
    # 获取文件路径对象
    ipath = Path(hello_world_f90)
    # 设置模拟的命令行参数，编译为 C 语言模块
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} -c'.split())

    # 在文件路径的父目录中执行测试
    with util.switchdir(ipath.parent):
        # 调用 f2pycli 函数
        f2pycli()
        # 构建命令行，运行 Python 代码以调用编译的模块
        cmd_run = shlex.split("python -c \"import blah; blah.hi()\"")
        # 运行命令，捕获输出
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        # 预期的输出
        eout = ' Hello World\n'
        # 断言运行结果的标准输出与预期输出一致
        assert rout.stdout == eout


# 函数定义：测试函数，用于验证 Numpy distutils 的编译器参数 --fcompiler
def test_npd_fcompiler():
    """
    CLI :: -c --fcompiler
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的编译器参数 --compiler
def test_npd_compiler():
    """
    CLI :: -c --compiler
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的帮助编译器参数 --help-fcompiler
def test_npd_help_fcompiler():
    """
    CLI :: -c --help-fcompiler
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的 Fortran 77 执行器参数 --f77exec
def test_npd_f77exec():
    """
    CLI :: -c --f77exec
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的 Fortran 90 执行器参数 --f90exec
def test_npd_f90exec():
    """
    CLI :: -c --f90exec
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的 Fortran 77 编译标志参数 --f77flags
def test_npd_f77flags():
    """
    CLI :: -c --f77flags
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的 Fortran 90 编译标志参数 --f90flags
def test_npd_f90flags():
    """
    CLI :: -c --f90flags
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的优化参数 --opt
def test_npd_opt():
    """
    CLI :: -c --opt
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的架构参数 --arch
def test_npd_arch():
    """
    CLI :: -c --arch
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的禁用优化参数 --noopt
def test_npd_noopt():
    """
    CLI :: -c --noopt
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的禁用架构参数 --noarch
def test_npd_noarch():
    """
    CLI :: -c --noarch
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的调试参数 --debug
def test_npd_debug():
    """
    CLI :: -c --debug
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的链接自动化参数 --link-<resource>
def test_npd_link_auto():
    """
    CLI :: -c --link-<resource>
    """
    # TODO: populate
    pass


# 函数定义：测试函数，用于验证 Numpy distutils 的库参数 --lib
def test_npd_lib():
    """
    CLI :: -c -L/path/to/lib/ -l<libname>
    """
    # 命令行接口（CLI）的说明，用法示例为 `-c -L/path/to/lib/ -l<libname>`
    # 在这里添加更多详细信息或操作说明
    pass
# 定义测试函数 `test_npd_define`，用于测试命令行接口中的 `-D<define>` 参数
def test_npd_define():
    """
    CLI :: -D<define>
    """
    # TODO: populate
    pass


# 定义测试函数 `test_npd_undefine`，用于测试命令行接口中的 `-U<name>` 参数
def test_npd_undefine():
    """
    CLI :: -U<name>
    """
    # TODO: populate
    pass


# 定义测试函数 `test_npd_incl`，用于测试命令行接口中的 `-I/path/to/include/` 参数
def test_npd_incl():
    """
    CLI :: -I/path/to/include/
    """
    # TODO: populate
    pass


# 定义测试函数 `test_npd_linker`，用于测试命令行接口中的 `<filename>.o <filename>.so <filename>.a` 参数
def test_npd_linker():
    """
    CLI :: <filename>.o <filename>.so <filename>.a
    """
    # TODO: populate
    pass
```