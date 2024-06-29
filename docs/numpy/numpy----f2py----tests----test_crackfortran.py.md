# `.\numpy\numpy\f2py\tests\test_crackfortran.py`

```
import importlib  # 导入标准库 importlib，用于动态加载模块
import codecs  # 导入标准库 codecs，提供编码和解码的工具函数
import time  # 导入标准库 time，提供时间相关的函数
import unicodedata  # 导入标准库 unicodedata，用于对 Unicode 字符进行数据库查询
import pytest  # 导入第三方库 pytest，用于编写和运行测试用例
import numpy as np  # 导入第三方库 numpy，并将其命名为 np，用于科学计算
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern  # 从 numpy.f2py.crackfortran 模块导入两个函数
from . import util  # 从当前包中导入 util 模块
from numpy.f2py import crackfortran  # 导入 numpy.f2py 模块中的 crackfortran 子模块
import textwrap  # 导入标准库 textwrap，用于简单的文本包装和填充
import contextlib  # 导入标准库 contextlib，用于创建和管理上下文对象
import io  # 导入标准库 io，提供了 Python 核心的基本 I/O 功能

class TestNoSpace(util.F2PyTest):
    # issue gh-15035: add handling for endsubroutine, endfunction with no space
    # between "end" and the block name
    sources = [util.getpath("tests", "src", "crackfortran", "gh15035.f")]

    def test_module(self):
        k = np.array([1, 2, 3], dtype=np.float64)  # 创建一个 numpy 数组 k，元素为 1, 2, 3，数据类型为 np.float64
        w = np.array([1, 2, 3], dtype=np.float64)  # 创建一个 numpy 数组 w，元素为 1, 2, 3，数据类型为 np.float64
        self.module.subb(k)  # 调用 self.module 对象的 subb 方法，传入参数 k
        assert np.allclose(k, w + 1)  # 使用 numpy 的 allclose 函数断言 k 是否与 w+1 全部近似相等
        self.module.subc([w, k])  # 调用 self.module 对象的 subc 方法，传入参数列表 [w, k]
        assert np.allclose(k, w + 1)  # 再次断言 k 是否与 w+1 全部近似相等
        assert self.module.t0("23") == b"2"  # 断言 self.module 对象的 t0 方法返回的结果是否等于 b"2"

class TestPublicPrivate:
    def test_defaultPrivate(self):
        fpath = util.getpath("tests", "src", "crackfortran", "privatemod.f90")  # 获取指定文件的路径
        mod = crackfortran.crackfortran([str(fpath)])  # 使用 crackfortran 模块的 crackfortran 函数处理文件路径
        assert len(mod) == 1  # 断言 mod 的长度是否为 1
        mod = mod[0]  # 获取 mod 的第一个元素
        assert "private" in mod["vars"]["a"]["attrspec"]  # 断言 mod 中的变量 a 的属性包含 "private"
        assert "public" not in mod["vars"]["a"]["attrspec"]  # 断言 mod 中的变量 a 的属性不包含 "public"
        assert "private" in mod["vars"]["b"]["attrspec"]  # 断言 mod 中的变量 b 的属性包含 "private"
        assert "public" not in mod["vars"]["b"]["attrspec"]  # 断言 mod 中的变量 b 的属性不包含 "public"
        assert "private" not in mod["vars"]["seta"]["attrspec"]  # 断言 mod 中的变量 seta 的属性不包含 "private"
        assert "public" in mod["vars"]["seta"]["attrspec"]  # 断言 mod 中的变量 seta 的属性包含 "public"

    def test_defaultPublic(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "publicmod.f90")  # 获取指定文件的路径
        mod = crackfortran.crackfortran([str(fpath)])  # 使用 crackfortran 模块的 crackfortran 函数处理文件路径
        assert len(mod) == 1  # 断言 mod 的长度是否为 1
        mod = mod[0]  # 获取 mod 的第一个元素
        assert "private" in mod["vars"]["a"]["attrspec"]  # 断言 mod 中的变量 a 的属性包含 "private"
        assert "public" not in mod["vars"]["a"]["attrspec"]  # 断言 mod 中的变量 a 的属性不包含 "public"
        assert "private" not in mod["vars"]["seta"]["attrspec"]  # 断言 mod 中的变量 seta 的属性不包含 "private"
        assert "public" in mod["vars"]["seta"]["attrspec"]  # 断言 mod 中的变量 seta 的属性包含 "public"

    def test_access_type(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "accesstype.f90")  # 获取指定文件的路径
        mod = crackfortran.crackfortran([str(fpath)])  # 使用 crackfortran 模块的 crackfortran 函数处理文件路径
        assert len(mod) == 1  # 断言 mod 的长度是否为 1
        tt = mod[0]['vars']  # 获取 mod 的第一个元素中的 'vars' 键对应的值
        assert set(tt['a']['attrspec']) == {'private', 'bind(c)'}  # 断言变量 a 的属性集合是否为 {'private', 'bind(c)'}
        assert set(tt['b_']['attrspec']) == {'public', 'bind(c)'}  # 断言变量 b_ 的属性集合是否为 {'public', 'bind(c)'}
        assert set(tt['c']['attrspec']) == {'public'}  # 断言变量 c 的属性集合是否为 {'public'}

    def test_nowrap_private_proceedures(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "gh23879.f90")  # 获取指定文件的路径
        mod = crackfortran.crackfortran([str(fpath)])  # 使用 crackfortran 模块的 crackfortran 函数处理文件路径
        assert len(mod) == 1  # 断言 mod 的长度是否为 1
        pyf = crackfortran.crack2fortran(mod)  # 使用 crackfortran 模块的 crack2fortran 函数处理 mod
        assert 'bar' not in pyf  # 断言字符串 'bar' 不在 pyf 中

class TestModuleProcedure():
    # TestModuleProcedure 类的定义暂无代码
    # 测试模块运算符的功能
    def test_moduleOperators(self, tmp_path):
        # 获取指定路径下的 operators.f90 文件路径
        fpath = util.getpath("tests", "src", "crackfortran", "operators.f90")
        # 使用 crackfortran 模块解析该文件，返回模块列表
        mod = crackfortran.crackfortran([str(fpath)])
        # 断言模块列表的长度为1
        assert len(mod) == 1
        # 取出第一个模块
        mod = mod[0]
        # 断言该模块包含键名为 "body"，并且其长度为9
        assert "body" in mod and len(mod["body"]) == 9
        # 断言模块的第二个元素的名称为 "operator(.item.)"
        assert mod["body"][1]["name"] == "operator(.item.)"
        # 断言第二个元素包含键名为 "implementedby"
        assert "implementedby" in mod["body"][1]
        # 断言第二个元素的 "implementedby" 值为 ["item_int", "item_real"]
        assert mod["body"][1]["implementedby"] == ["item_int", "item_real"]
        # 断言模块的第三个元素的名称为 "operator(==)"
        assert mod["body"][2]["name"] == "operator(==)"
        # 断言第三个元素包含键名为 "implementedby"
        assert "implementedby" in mod["body"][2]
        # 断言第三个元素的 "implementedby" 值为 ["items_are_equal"]
        assert mod["body"][2]["implementedby"] == ["items_are_equal"]
        # 断言模块的第四个元素的名称为 "assignment(=)"
        assert mod["body"][3]["name"] == "assignment(=)"
        # 断言第四个元素包含键名为 "implementedby"
        assert "implementedby" in mod["body"][3]
        # 断言第四个元素的 "implementedby" 值为 ["get_int", "get_real"]
    
    # 测试模块的非公有（private）和公有（public）属性设置
    def test_notPublicPrivate(self, tmp_path):
        # 获取指定路径下的 pubprivmod.f90 文件路径
        fpath = util.getpath("tests", "src", "crackfortran", "pubprivmod.f90")
        # 使用 crackfortran 模块解析该文件，返回模块列表
        mod = crackfortran.crackfortran([str(fpath)])
        # 断言模块列表的长度为1
        assert len(mod) == 1
        # 取出第一个模块
        mod = mod[0]
        # 断言模块变量 'a' 的 'attrspec' 属性为 ['private', ]
        assert mod['vars']['a']['attrspec'] == ['private', ]
        # 断言模块变量 'b' 的 'attrspec' 属性为 ['public', ]
        assert mod['vars']['b']['attrspec'] == ['public', ]
        # 断言模块变量 'seta' 的 'attrspec' 属性为 ['public', ]
        assert mod['vars']['seta']['attrspec'] == ['public', ]
class TestExternal(util.F2PyTest):
    # 问题编号 gh-17859: 添加对外部属性的支持
    sources = [util.getpath("tests", "src", "crackfortran", "gh17859.f")]

    def test_external_as_statement(self):
        # 定义一个简单的增加函数
        def incr(x):
            return x + 123

        # 调用被测试模块中的 external_as_statement 方法，并验证返回结果
        r = self.module.external_as_statement(incr)
        assert r == 123

    def test_external_as_attribute(self):
        # 定义一个简单的增加函数
        def incr(x):
            return x + 123

        # 调用被测试模块中的 external_as_attribute 方法，并验证返回结果
        r = self.module.external_as_attribute(incr)
        assert r == 123


class TestCrackFortran(util.F2PyTest):
    # 问题编号 gh-2848: 在 Fortran 子程序参数列表中的参数之间添加注释行
    sources = [util.getpath("tests", "src", "crackfortran", "gh2848.f90")]

    def test_gh2848(self):
        # 调用被测试模块中的 gh2848 方法，并验证返回结果
        r = self.module.gh2848(1, 2)
        assert r == (1, 2)


class TestMarkinnerspaces:
    # 问题编号 gh-14118: markinnerspaces 不处理多重引号

    def test_do_not_touch_normal_spaces(self):
        # 针对普通字符串，验证 markinnerspaces 函数不做修改
        test_list = ["a ", " a", "a b c", "'abcdefghij'"]
        for i in test_list:
            assert markinnerspaces(i) == i

    def test_one_relevant_space(self):
        # 针对带有一个有意义空格的字符串，验证 markinnerspaces 函数替换空格
        assert markinnerspaces("a 'b c' \\' \\'") == "a 'b@_@c' \\' \\'"
        assert markinnerspaces(r'a "b c" \" \"') == r'a "b@_@c" \" \"'

    def test_ignore_inner_quotes(self):
        # 针对带有内部引号的字符串，验证 markinnerspaces 函数只处理外部空格
        assert markinnerspaces("a 'b c\" \" d' e") == "a 'b@_@c\"@_@\"@_@d' e"
        assert markinnerspaces("a \"b c' ' d\" e") == "a \"b@_@c'@_@'@_@d\" e"

    def test_multiple_relevant_spaces(self):
        # 针对带有多个有意义空格的字符串，验证 markinnerspaces 函数替换空格
        assert markinnerspaces("a 'b c' 'd e'") == "a 'b@_@c' 'd@_@e'"
        assert markinnerspaces(r'a "b c" "d e"') == r'a "b@_@c" "d@_@e"'


class TestDimSpec(util.F2PyTest):
    """This test suite tests various expressions that are used as dimension
    specifications.

    There exists two usage cases where analyzing dimensions
    specifications are important.

    In the first case, the size of output arrays must be defined based
    on the inputs to a Fortran function. Because Fortran supports
    arbitrary bases for indexing, for instance, `arr(lower:upper)`,
    f2py has to evaluate an expression `upper - lower + 1` where
    `lower` and `upper` are arbitrary expressions of input parameters.
    The evaluation is performed in C, so f2py has to translate Fortran
    expressions to valid C expressions (an alternative approach is
    that a developer specifies the corresponding C expressions in a
    .pyf file).

    In the second case, when user provides an input array with a given
    size but some hidden parameters used in dimensions specifications
    need to be determined based on the input array size. This is a
    harder problem because f2py has to solve the inverse problem: find
    a parameter `p` such that `upper(p) - lower(p) + 1` equals to the
    size of input array. In the case when this equation cannot be
    solved (e.g. because the input array size is wrong), raise an
    error before calling the Fortran function (that otherwise would
    """
    # 定义文件名后缀为 ".f90"
    suffix = ".f90"

    # 定义代码模板，使用 textwrap.dedent 去除代码缩进
    code_template = textwrap.dedent("""
      function get_arr_size_{count}(a, n) result (length)
        integer, intent(in) :: n
        integer, dimension({dimspec}), intent(out) :: a
        integer length
        length = size(a)
      end function

      subroutine get_inv_arr_size_{count}(a, n)
        integer :: n
        ! the value of n is computed in f2py wrapper
        !f2py intent(out) n
        integer, dimension({dimspec}), intent(in) :: a
        if (a({first}).gt.0) then
          ! print*, "a=", a
        endif
      end subroutine
    """)

    # 线性维度规格列表
    linear_dimspecs = [
        "n", "2*n", "2:n", "n/2", "5 - n/2", "3*n:20", "n*(n+1):n*(n+5)",
        "2*n, n"
    ]
    
    # 非线性维度规格列表
    nonlinear_dimspecs = ["2*n:3*n*n+2*n"]
    
    # 所有维度规格列表，包括线性和非线性
    all_dimspecs = linear_dimspecs + nonlinear_dimspecs

    # 初始化代码字符串
    code = ""
    
    # 遍历所有维度规格
    for count, dimspec in enumerate(all_dimspecs):
        # 获取维度规格中的起始值列表
        lst = [(d.split(":")[0] if ":" in d else "1") for d in dimspec.split(',')]
        # 使用代码模板填充代码字符串
        code += code_template.format(
            count=count,
            dimspec=dimspec,
            first=", ".join(lst),
        )

    # 使用 pytest.mark.parametrize 标记测试参数化，参数为所有维度规格
    @pytest.mark.parametrize("dimspec", all_dimspecs)
    @pytest.mark.slow
    def test_array_size(self, dimspec):
        # 获取当前维度规格在列表中的索引
        count = self.all_dimspecs.index(dimspec)
        # 获取相应的 get_arr_size 函数
        get_arr_size = getattr(self.module, f"get_arr_size_{count}")

        # 遍历测试用例中的不同 n 值
        for n in [1, 2, 3, 4, 5]:
            # 调用 get_arr_size 函数获取返回值 sz 和数组 a
            sz, a = get_arr_size(n)
            # 断言数组 a 的大小与返回的 sz 相等
            assert a.size == sz

    # 使用 pytest.mark.parametrize 标记测试参数化，参数为所有维度规格
    @pytest.mark.parametrize("dimspec", all_dimspecs)
    def test_inv_array_size(self, dimspec):
        # 获取当前维度规格在列表中的索引
        count = self.all_dimspecs.index(dimspec)
        # 获取相应的 get_arr_size 和 get_inv_arr_size 函数
        get_arr_size = getattr(self.module, f"get_arr_size_{count}")
        get_inv_arr_size = getattr(self.module, f"get_inv_arr_size_{count}")

        # 遍历测试用例中的不同 n 值
        for n in [1, 2, 3, 4, 5]:
            # 调用 get_arr_size 函数获取返回值 sz 和数组 a
            sz, a = get_arr_size(n)
            
            # 如果当前维度规格在非线性维度规格列表中
            if dimspec in self.nonlinear_dimspecs:
                # 调用 get_inv_arr_size 函数，需要指定 n 作为输入
                n1 = get_inv_arr_size(a, n)
            else:
                # 否则，在线性依赖的情况下，n 可以从数组的形状中确定
                n1 = get_inv_arr_size(a)
            
            # 断言返回的 n1 值经过处理后得到的数组 sz1 与原始 sz 相等
            sz1, _ = get_arr_size(n1)
            assert sz == sz1, (n, n1, sz, sz1)
class TestModuleDeclaration:
    # 测试模块声明类
    def test_dependencies(self, tmp_path):
        # 获取测试文件路径
        fpath = util.getpath("tests", "src", "crackfortran", "foo_deps.f90")
        # 调用 crackfortran 函数处理文件路径并返回模块列表
        mod = crackfortran.crackfortran([str(fpath)])
        # 断言模块列表长度为1
        assert len(mod) == 1
        # 断言模块中第一个元素的变量 'abar' 的赋值为 "bar('abar')"
        assert mod[0]["vars"]["abar"]["="] == "bar('abar')"


class TestEval(util.F2PyTest):
    # 测试 _eval_scalar 函数
    def test_eval_scalar(self):
        eval_scalar = crackfortran._eval_scalar

        # 断言将字符串 '123' 传入 eval_scalar 函数返回 '123'
        assert eval_scalar('123', {}) == '123'
        # 断言将字符串 '12 + 3' 传入 eval_scalar 函数返回 '15'
        assert eval_scalar('12 + 3', {}) == '15'
        # 断言将字符串 'a + b' 和字典 {'a': 1, 'b': 2} 传入 eval_scalar 函数返回 '3'
        assert eval_scalar('a + b', dict(a=1, b=2)) == '3'
        # 断言将字符串 '"123"' 传入 eval_scalar 函数返回 "'123'"
        assert eval_scalar('"123"', {}) == "'123'"


class TestFortranReader(util.F2PyTest):
    # 测试 Fortran 读取器类
    @pytest.mark.parametrize("encoding",
                             ['ascii', 'utf-8', 'utf-16', 'utf-32'])
    def test_input_encoding(self, tmp_path, encoding):
        # 为了解决 gh-635 的问题，创建带有指定编码的临时文件
        f_path = tmp_path / f"input_with_{encoding}_encoding.f90"
        with f_path.open('w', encoding=encoding) as ff:
            # 向文件写入 Fortran 子例程定义
            ff.write("""
                     subroutine foo()
                     end subroutine foo
                     """)
        # 使用 crackfortran 函数处理文件路径并返回模块列表
        mod = crackfortran.crackfortran([str(f_path)])
        # 断言模块列表中第一个模块的名称为 'foo'
        assert mod[0]['name'] == 'foo'


@pytest.mark.slow
class TestUnicodeComment(util.F2PyTest):
    # 测试 Unicode 注释处理类
    sources = [util.getpath("tests", "src", "crackfortran", "unicode_comment.f90")]

    @pytest.mark.skipif(
        (importlib.util.find_spec("charset_normalizer") is None),
        reason="test requires charset_normalizer which is not installed",
    )
    def test_encoding_comment(self):
        # 调用模块中的 foo 方法，传入参数 3
        self.module.foo(3)


class TestNameArgsPatternBacktracking:
    # 测试名称参数模式回溯类
    @pytest.mark.parametrize(
        ['adversary'],
        [
            ('@)@bind@(@',),
            ('@)@bind                         @(@',),
            ('@)@bind foo bar baz@(@',)
        ]
    )
    # 定义一个测试函数，用于检测名为 `nameargspattern` 的正则表达式在处理 `adversary` 字符串时的反向递归问题
    def test_nameargspattern_backtracking(self, adversary):
        '''address ReDOS vulnerability:
        https://github.com/numpy/numpy/issues/23338'''
        
        # 每批次测试的次数
        trials_per_batch = 12
        # 每个正则表达式的批次数
        batches_per_regex = 4
        # 重复拷贝 `adversary` 字符串的次数范围
        start_reps, end_reps = 15, 25
        
        # 循环遍历重复次数范围内的每个值
        for ii in range(start_reps, end_reps):
            # 构造重复次数 `ii` 倍的 `adversary` 字符串
            repeated_adversary = adversary * ii
            
            # 在小批次中多次测试，以增加捕获不良正则表达式的机会
            for _ in range(batches_per_regex):
                times = []
                
                # 在每个批次中多次运行测试
                for _ in range(trials_per_batch):
                    # 记录测试开始时间
                    t0 = time.perf_counter()
                    # 在 `repeated_adversary` 中搜索 `nameargspattern` 正则表达式
                    mtch = nameargspattern.search(repeated_adversary)
                    # 计算测试所用时间并记录
                    times.append(time.perf_counter() - t0)
                
                # 断言：正则表达式的性能应该远快于每次搜索耗时超过 0.2 秒
                assert np.median(times) < 0.2
            
            # 断言：不应该找到匹配，即 `mtch` 应为 None
            assert not mtch
            
            # 构造一个含有 '@)@' 后缀的 `repeated_adversary` 字符串，检测是否可以通过旧版本的正则表达式
            good_version_of_adversary = repeated_adversary + '@)@'
            # 断言：应该能够找到匹配，即 `nameargspattern` 应该能够匹配 `good_version_of_adversary`
            assert nameargspattern.search(good_version_of_adversary)
class TestFunctionReturn(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "gh23598.f90")]

    @pytest.mark.slow
    def test_function_rettype(self):
        # 标记此测试为缓慢执行，验证函数返回类型
        # gh-23598
        assert self.module.intproduct(3, 4) == 12


class TestFortranGroupCounters(util.F2PyTest):
    def test_end_if_comment(self):
        # gh-23533
        # 准备测试gh-23533文件的路径
        fpath = util.getpath("tests", "src", "crackfortran", "gh23533.f")
        try:
            # 尝试运行crackfortran.crackfortran来解析Fortran文件
            crackfortran.crackfortran([str(fpath)])
        except Exception as exc:
            # 如果抛出异常，则断言失败，显示异常信息
            assert False, f"'crackfortran.crackfortran' raised an exception {exc}"


class TestF77CommonBlockReader():
    def test_gh22648(self, tmp_path):
        # 准备测试gh-22648文件的路径
        fpath = util.getpath("tests", "src", "crackfortran", "gh22648.pyf")
        with contextlib.redirect_stdout(io.StringIO()) as stdout_f2py:
            # 运行crackfortran.crackfortran来解析Fortran文件，并重定向标准输出
            mod = crackfortran.crackfortran([str(fpath)])
        # 断言标准输出中不包含"Mismatch"
        assert "Mismatch" not in stdout_f2py.getvalue()

class TestParamEval():
    # issue gh-11612, array parameter parsing
    def test_param_eval_nested(self):
        # 准备测试参数解析，包含嵌套结构
        v = '(/3.14, 4./)'
        g_params = dict(kind=crackfortran._kind_func,
                selected_int_kind=crackfortran._selected_int_kind_func,
                selected_real_kind=crackfortran._selected_real_kind_func)
        params = {'dp': 8, 'intparamarray': {1: 3, 2: 5},
                  'nested': {1: 1, 2: 2, 3: 3}}
        dimspec = '(2)'
        # 调用crackfortran.param_eval进行参数解析，验证结果是否符合预期
        ret = crackfortran.param_eval(v, g_params, params, dimspec=dimspec)
        assert ret == {1: 3.14, 2: 4.0}

    def test_param_eval_nonstandard_range(self):
        # 准备测试非标准范围的参数解析
        v = '(/ 6, 3, 1 /)'
        g_params = dict(kind=crackfortran._kind_func,
                selected_int_kind=crackfortran._selected_int_kind_func,
                selected_real_kind=crackfortran._selected_real_kind_func)
        params = {}
        dimspec = '(-1:1)'
        # 调用crackfortran.param_eval进行参数解析，验证结果是否符合预期
        ret = crackfortran.param_eval(v, g_params, params, dimspec=dimspec)
        assert ret == {-1: 6, 0: 3, 1: 1}

    def test_param_eval_empty_range(self):
        # 准备测试空范围的参数解析
        v = '6'
        g_params = dict(kind=crackfortran._kind_func,
                selected_int_kind=crackfortran._selected_int_kind_func,
                selected_real_kind=crackfortran._selected_real_kind_func)
        params = {}
        dimspec = ''
        # 检查传递空范围参数时，是否抛出ValueError异常
        pytest.raises(ValueError, crackfortran.param_eval, v, g_params, params,
                      dimspec=dimspec)

    def test_param_eval_non_array_param(self):
        # 准备测试非数组参数的解析
        v = '3.14_dp'
        g_params = dict(kind=crackfortran._kind_func,
                selected_int_kind=crackfortran._selected_int_kind_func,
                selected_real_kind=crackfortran._selected_real_kind_func)
        params = {}
        # 调用crackfortran.param_eval进行参数解析，验证结果是否符合预期
        ret = crackfortran.param_eval(v, g_params, params, dimspec=None)
        assert ret == '3.14_dp'
    # 定义一个测试方法，用于测试参数评估是否具有过多维度
    def test_param_eval_too_many_dims(self):
        # 定义一个包含 Fortran 样式重塑函数的字符串参数
        v = 'reshape((/ (i, i=1, 250) /), (/5, 10, 5/))'
        # 创建一个全局参数字典，包括三个函数作为值
        g_params = dict(kind=crackfortran._kind_func,
                        selected_int_kind=crackfortran._selected_int_kind_func,
                        selected_real_kind=crackfortran._selected_real_kind_func)
        # 初始化一个空参数字典
        params = {}
        # 定义维度规范字符串
        dimspec = '(0:4, 3:12, 5)'
        # 使用 pytest 模块断言函数引发 ValueError 异常，调用 crackfortran.param_eval 函数进行测试
        pytest.raises(ValueError, crackfortran.param_eval, v, g_params, params,
                      dimspec=dimspec)
```