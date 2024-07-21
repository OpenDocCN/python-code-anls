# `.\pytorch\test\dynamo\test_bytecode_utils.py`

```py
# Owner(s): ["module: dynamo"]

import collections  # 导入collections模块，用于处理容器数据类型
import dis  # 导入dis模块，用于反汇编Python字节码
import sys  # 导入sys模块，提供访问与Python解释器相关的变量和函数
import unittest  # 导入unittest模块，用于编写和运行单元测试

import torch  # 导入torch库，用于深度学习任务
import torch._dynamo.test_case  # 导入torch._dynamo.test_case模块，可能是测试框架的一部分
from torch._dynamo import bytecode_analysis, bytecode_transformation  # 导入字节码分析和转换相关的模块
from torch._dynamo.testing import skipIfNotPy311, skipIfNotPy312  # 导入条件跳过测试的装饰器函数


class BytecodeTests(torch._dynamo.test_case.TestCase):
    @skipIfNotPy311  # 如果Python版本小于3.11，则跳过测试
    def test_linetable_311_writer1(self):
        def fn():
            a = 10  # 定义变量a为整数10
            b = 20  # 定义变量b为整数20
            c = a + b  # 计算a和b的和，赋值给变量c
            f = "linetable_writer"  # 定义字符串变量f为'linetable_writer'
            return f"Test if {f} generates correct co_linetable: {c}"  # 返回格式化字符串

        keys = bytecode_transformation.get_code_keys()  # 调用函数获取字节码键的集合
        code_options = {k: getattr(fn.__code__, k) for k in keys}  # 构建字典，存储函数fn字节码属性
        result = bytecode_transformation.clean_and_assemble_instructions(
            bytecode_transformation.cleaned_instructions(fn.__code__),  # 清理并组装函数fn的字节码指令
            keys,  # 使用的字节码键集合
            code_options,  # 字节码选项字典
        )
        l1, l2 = list(fn.__code__.co_positions()), list(result[1].co_positions())  # 获取函数fn和结果中字节码位置的列表
        self.assertEqual(len(l1), len(l2))  # 断言两个位置列表的长度相等
        for p1, p2 in zip(l1, l2):
            self.assertEqual(p1, p2)  # 逐个比较位置列表中的元素
        # TODO co_lnotab is deprecated in 3.12 and will be removed in 3.14
        # In 3.11+,. it is computed lazily from other linetable attributes (e.g. co_linetable),
        # so we do not set this attribute ourselves.
        self.assertEqual(fn.__code__.co_lnotab, result[1].co_lnotab)  # 断言函数fn的co_lnotab属性与结果中的相等

    @skipIfNotPy311  # 如果Python版本小于3.11，则跳过测试
    def test_linetable_311_writer2(self):
        """
        test large ops (LOAD_METHOD) and EXTENDED_ARGS
        fn_str is in the form:
        def fn():
            ...
            x0 = 1
            x1 = 1
            ...
            l = [x0, x1, ...]
        """
        fn_str = f"""\
def fn():
    foo.bar(1, 2, 3)
{str(chr(10)).join(' ' * 4 + 'x' + str(i) + ' = 1' for i in range(1 << 9))}
    l = [{' '.join('x' + str(i) + ',' for i in range(1 << 9))}]
        """
        locals = {}  # 创建一个空字典locals
        exec(fn_str, {}, locals)  # 执行动态生成的函数fn_str，并将局部变量存储在locals中
        fn = locals["fn"]  # 获取动态生成的函数fn
        orig_inst_str = "\n".join(list(map(str, dis.get_instructions(fn))))  # 获取原始函数fn的指令字符串
        self.assertIn("EXTENDED_ARG", orig_inst_str)  # 断言原始指令字符串中包含"EXTENDED_ARG"
        load_method_str = "LOAD_ATTR" if sys.version_info >= (3, 12) else "LOAD_METHOD"
        self.assertIn(load_method_str, orig_inst_str)  # 根据Python版本断言原始指令字符串中包含"LOAD_ATTR"或"LOAD_METHOD"
        keys = bytecode_transformation.get_code_keys()  # 获取字节码键集合
        code_options = {k: getattr(fn.__code__, k) for k in keys}  # 构建字典，存储函数fn字节码属性
        result = bytecode_transformation.clean_and_assemble_instructions(
            bytecode_transformation.cleaned_instructions(fn.__code__),  # 清理并组装函数fn的字节码指令
            keys,  # 使用的字节码键集合
            code_options,  # 字节码选项字典
        )
        new_inst_str = "\n".join(list(map(str, result[0])))  # 获取清理后的指令字符串
        self.assertIn("EXTENDED_ARG", new_inst_str)  # 断言清理后的指令字符串中包含"EXTENDED_ARG"
        self.assertIn(load_method_str, new_inst_str)  # 根据Python版本断言清理后的指令字符串中包含"LOAD_ATTR"或"LOAD_METHOD"
        l1, l2 = list(fn.__code__.co_positions()), list(result[1].co_positions())  # 获取函数fn和结果中字节码位置的列表
        self.assertEqual(len(l1), len(l2))  # 断言两个位置列表的长度相等
        for p1, p2 in zip(l1, l2):
            self.assertEqual(p1, p2)  # 逐个比较位置列表中的元素
        self.assertEqual(fn.__code__.co_lnotab, result[1].co_lnotab)  # 断言函数fn的co_lnotab属性与结果中的相等
    @unittest.skipIf(
        sys.version_info < (3, 10) or sys.version_info >= (3, 11),
        "linetable test for Python 3.10",
    )
    # 定义一个测试函数，用于测试在 Python 3.10 版本下生成正确的 co_linetable
    def test_linetable_310_writer(self):
        # 定义一个简单的函数 fn
        def fn():
            # 设置变量 a 和 b 的值
            a = 10
            b = 20
            # 计算 c 的值为 a + b
            c = a + b
            # 设置字符串变量 f 的值为 "linetable_writer"
            f = "linetable_writer"
            # 返回一个包含 f 和 c 的字符串，用于测试生成正确的 co_linetable
            return f"Test if {f} generates correct co_linetable: {c}"

        # 使用 dis 模块获取 fn 函数的指令集合
        inst = dis.get_instructions(fn)
        # 调用 bytecode_transformation 模块的 assemble 函数来组装指令，并传入函数 fn 的第一行代码的行号
        result = bytecode_transformation.assemble(inst, fn.__code__.co_firstlineno)
        # 断言测试结果中的第二个元素与 fn 函数对象的 co_linetable 相等
        self.assertTrue(result[1] == fn.__code__.co_linetable)

    @unittest.skipIf(sys.version_info >= (3, 10), "use lnotab when python < 3.10")
    # 定义一个测试函数，用于在 Python 版本小于 3.10 时使用 co_lnotab
    def test_lnotab_writer(self):
        # 定义一个简单的函数 fn
        def fn():
            # 设置变量 a 和 b 的值
            a = 10
            b = 20
            # 计算 c 的值为 a + b
            c = a + b
            # 设置字符串变量 f 的值为 "lnotab_writer"
            f = "lnotab_writer"
            # 返回一个包含 f 和 c 的字符串，用于测试生成正确的 co_lnotab
            return f"Test if {f} generates correct co_lnotab: {c}"

        # 使用 dis 模块获取 fn 函数的指令集合
        inst = dis.get_instructions(fn)
        # 调用 bytecode_transformation 模块的 assemble 函数来组装指令，并传入函数 fn 的第一行代码的行号
        result = bytecode_transformation.assemble(inst, fn.__code__.co_firstlineno)
        # 断言测试结果中的第二个元素与 fn 函数对象的 co_lnotab 相等
        self.assertTrue(result[1] == fn.__code__.co_lnotab)

    # 定义一个测试函数，用于测试如果张量为 None 时的情况
    def test_if_tensor_is_none(self):
        """
        Python 3.11 adds new jump instructions that check if
        TOS is None. We do not support these instructions.
        """
        # 定义一个简单的函数 f，接受 x 和 y 作为参数
        def f(x, y):
            # 设置变量 z 的初始值为 1
            z = 1
            # 如果 x 是 None，则 z 乘以 2
            if x is None:
                z *= 2
            # 如果 y 不是 None，则 z 乘以 3
            if y is not None:
                z *= 3
            # 返回变量 z 的值
            return z

        # 使用 torch._dynamo.optimize 进行函数 f 的优化
        opt_f = torch._dynamo.optimize("eager", nopython=True)(f)
        # 断言优化后的函数 opt_f 返回值与预期结果相等
        self.assertEqual(opt_f(None, torch.ones(2)), 6)

        # 如果 Python 版本大于等于 3.11，则进行以下操作
        if sys.version_info >= (3, 11):
            # 使用 bytecode_transformation 模块的 cleaned_instructions 函数获取函数 f 的指令集合
            insts = bytecode_transformation.cleaned_instructions(f.__code__)
            # 遍历指令集合中的每个指令
            for inst in insts:
                # 断言指令的操作名称中不包含 "_NONE"
                self.assertNotIn("_NONE", inst.opname)

    @skipIfNotPy311
    # 定义一个测试函数，用于测试异常表编码的变长整数
    def test_exception_table_encode_varint(self):
        # 定义一个包含两个二进制数值的列表 nums
        nums = [
            0b111_101010_000000,
            0b1100_111000_010101_101010,
        ]
        # 调用 bytecode_transformation 模块的 encode_exception_table_varint 函数对 nums[0] 进行编码，并连接上 nums[1] 的编码结果
        b = bytecode_transformation.encode_exception_table_varint(
            nums[0]
        ) + bytecode_transformation.encode_exception_table_varint(nums[1])
        # 创建一个空列表 nums_new，用于存储解码后的整数值
        nums_new = []
        # 创建一个字节迭代器 b_iter
        b_iter = iter(bytes(b))
        # 不断循环直到捕获 StopIteration 异常
        while True:
            try:
                # 将解码后的整数值添加到 nums_new 列表中
                nums_new.append(
                    bytecode_transformation.decode_exception_table_varint(b_iter)
                )
            except StopIteration:
                # 捕获到 StopIteration 异常时退出循环
                break
        # 断言 nums_new 列表与原始的 nums 列表相等
        self.assertEqual(nums, nums_new)

    @skipIfNotPy311
    # 定义一个测试函数，用于测试异常表的解析
    def test_exception_table_parsing(self):
        # 定义一个包含异常处理的函数 fn
        def fn():
            try:
                with a():
                    b()
                c()
            except Exception:
                d()
            finally:
                e()
            f()

        # 使用 bytecode_transformation 模块的 parse_exception_table 函数解析 fn 函数对象的 co_exceptiontable 属性
        tab = bytecode_transformation.parse_exception_table(
            fn.__code__.co_exceptiontable
        )
        # 使用 bytecode_transformation 模块的 assemble_exception_table 函数组装异常表
        b = bytecode_transformation.assemble_exception_table(tab)
        # 断言组装后的异常表与 fn 函数对象的 co_exceptiontable 属性相等
        self.assertEqual(b, fn.__code__.co_exceptiontable)
    def test_exception_table_e2e(self):
        # 定义一个测试函数 fn，用于测试异常处理和 finally 块的执行顺序
        def fn():
            try:
                # 尝试执行上下文管理器 a()，并在其内部调用 b()
                with a():
                    b()
                # 如果没有异常，则继续执行 c()
                c()
            except Exception:
                # 捕获任何异常，并执行 d()
                d()
            finally:
                # 无论是否发生异常，最终执行 e()
                e()
            # 最后执行 f()
            f()

        # 定义一个什么都不做的函数 nothing
        def nothing(*args):
            pass

        # 对 fn 函数的字节码进行转换，应用 nothing 函数
        code = bytecode_transformation.transform_code_object(fn.__code__, nothing)
        # 断言转换后的字节码的异常处理表与原始函数 fn 的一致
        self.assertEqual(code.co_exceptiontable, fn.__code__.co_exceptiontable)

    @skipIfNotPy311
    def test_exception_table_e2e_2(self):
        # 定义一个测试函数 fn，仅包含一个异常处理块
        def fn():
            try:
                # 尝试返回变量 a
                return a
            except Exception:
                # 捕获任何异常，什么也不做
                pass

        # 定义一个什么都不做的函数 nothing
        def nothing(*args):
            pass

        # 对 fn 函数的字节码进行转换，应用 nothing 函数
        code = bytecode_transformation.transform_code_object(fn.__code__, nothing)
        # 断言转换后的字节码的异常处理表与原始函数 fn 的一致
        self.assertEqual(code.co_exceptiontable, fn.__code__.co_exceptiontable)

    @skipIfNotPy311
    def test_exception_table_entry_propagation(self):
        # 创建一个指令列表 insts，包含 10 个 "NOP" 指令
        insts = []
        for _ in range(10):
            insts.append(bytecode_transformation.create_instruction("NOP"))
        
        # 设置不同指令的异常处理表条目，通过 propagate_inst_exn_table_entries 函数传播
        insts[8].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[0], insts[9], insts[0], 0, True
        )
        insts[0].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[0], insts[0], insts[1], 0, True
        )
        insts[1].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[0], insts[2], insts[2], 0, True
        )
        insts[5].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[4], insts[6], insts[3], 0, True
        )
        insts[9].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[9], insts[9], insts[4], 0, True
        )
        insts[7].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[7], insts[9], insts[5], 0, True
        )
        
        # 传播异常处理表条目
        bytecode_transformation.propagate_inst_exn_table_entries(insts)
        
        # 定义预期的结果列表
        expected = [1, 2, 2, 0, 3, 3, 3, 5, 5, 4]
        
        # 遍历 insts 和 expected 列表，断言每个指令的异常处理表条目的正确性
        for inst, exp in zip(insts, expected):
            self.assertIsNotNone(inst.exn_tab_entry)
            self.assertIs(inst.exn_tab_entry.target, insts[exp])
    # 定义一个测试方法，用于测试异常表的嵌套计算
    def test_compute_exception_table_nested(self):
        # 创建一个空列表来存放指令
        insts = []
        # 循环20次，向指令列表中添加“NOP”指令对象
        for _ in range(20):
            insts.append(bytecode_transformation.create_instruction("NOP"))
        
        # 设置第10个指令的异常表项
        insts[10].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[1], insts[10], insts[0], 0, True
        )
        
        # 设置第0个指令的异常表项
        insts[0].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[1], insts[1], insts[1], 0, True
        )
        
        # 设置第1个指令的异常表项
        insts[1].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[1], insts[3], insts[2], 0, True
        )
        
        # 设置第5个指令的异常表项
        insts[5].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[5], insts[7], insts[3], 0, True
        )
        
        # 设置第9个指令的异常表项
        insts[9].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[10], insts[10], insts[4], 0, True
        )
        
        # 设置第7个指令的异常表项
        insts[7].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[8], insts[10], insts[5], 0, True
        )
        
        # 设置第14个指令的异常表项
        insts[14].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[13], insts[17], insts[6], 0, True
        )
        
        # 设置第16个指令的异常表项
        insts[16].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[15], insts[16], insts[7], 0, True
        )
        
        # 更新指令的偏移量
        bytecode_transformation.update_offsets(insts)
        
        # 计算指令列表的异常表
        tab = bytecode_transformation.compute_exception_table(insts)
        
        # 预期的异常表项列表
        expected = [
            (1, 1, 1),
            (2, 3, 2),
            (4, 4, 0),
            (5, 7, 3),
            (8, 9, 5),
            (10, 10, 4),
            (13, 14, 6),
            (15, 16, 7),
            (17, 17, 6),
        ]
        
        # 断言实际生成的异常表与预期的异常表相等
        self.assertEqual(len(tab), len(expected))
        
        # 逐个比较每个异常表项的起始位置、结束位置和目标位置是否符合预期
        for entry, exp in zip(tab, expected):
            self.assertEqual(entry.start, exp[0] * 2)
            self.assertEqual(entry.end, exp[1] * 2)
            self.assertEqual(entry.target, exp[2] * 2)

    @skipIfNotPy311
    def test_remove_dead_code_with_exn_table_entries(self):
        # 导入函数：创建指令
        create_instruction = bytecode_transformation.create_instruction
        # 创建四个NOP指令作为目标指令
        target1 = create_instruction("NOP")
        target2 = create_instruction("NOP")
        target3 = create_instruction("NOP")
        # 创建一个NOP指令作为异常处理的起始点
        exn_start = create_instruction("NOP")
        # 创建一个NOP指令作为异常处理的结束点
        exn_end = create_instruction("NOP")
        # 创建指令列表，包含不同类型的指令及目标指令
        insts = [
            create_instruction("JUMP_FORWARD", target=target1),
            exn_start,  # 此处的指令被认为是无效的（dead）
            target1,
            create_instruction("JUMP_FORWARD", target=target3),
            exn_end,  # 此处的指令被认为是无效的（dead）
            target2,
            target3,
        ]
        # 为起始异常处理指令添加异常表条目
        exn_start.exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            exn_start, exn_end, target2, 0, True
        )
        # 传播指令的异常表条目
        bytecode_transformation.propagate_inst_exn_table_entries(insts)
        # 移除死代码
        insts = bytecode_analysis.remove_dead_code(insts)
        # 断言指令列表长度为5
        self.assertEqual(len(insts), 5)
        # 确保异常处理的起始指令不在指令列表中
        self.assertNotIn(exn_start, insts)
        # 确保异常处理的结束指令不在指令列表中
        self.assertNotIn(exn_end, insts)
        # 确保目标指令2在指令列表中
        self.assertIn(target2, insts)
        # 确保目标指令3在指令列表中
        self.assertIn(target3, insts)
        # 更新指令的偏移量
        bytecode_transformation.update_offsets(insts)
        # 计算异常表
        tab = bytecode_transformation.compute_exception_table(insts)
        # 断言异常表长度为1
        self.assertEqual(len(tab), 1)
        # 确保异常表的起始位置为2
        self.assertEqual(tab[0].start, 2)
        # 确保异常表的结束位置为4
        self.assertEqual(tab[0].end, 4)
        # 确保异常表的目标位置为6
        self.assertEqual(tab[0].target, 6)

    def test_bytecode_from_template(self):
        # 定义一个函数fn，接收一个字典参数d1
        def fn(d1):
            # 遍历d1中的键值对，将其赋值给d2
            for k, v in d1.items():
                d2[k] = v

        # 定义变量名映射字典
        varname_map = {"d1": "var1", "d2": "var2", "k": "var3", "v": "var4"}
        # 使用模板生成字节码指令列表
        insts = bytecode_transformation.bytecode_from_template(fn, varname_map)
        # 遍历生成的指令列表
        for inst in insts:
            # 确保指令的起始行为None
            self.assertIsNone(inst.starts_line)
            # 如果指令操作名称以"LOAD"开头
            if inst.opname.startswith("LOAD"):
                # 确保指令的参数值不在变量名映射字典中
                self.assertNotIn(inst.argval, varname_map)
                # 如果指令操作名称不是"LOAD_GLOBAL"和"LOAD_ATTR"
                if inst.opname not in ("LOAD_GLOBAL", "LOAD_ATTR"):
                    # 确保指令的参数为None
                    self.assertIsNone(inst.arg)
            # 确保指令操作名称不以"RETURN"开头
            self.assertFalse(inst.opname.startswith("RETURN"))

    @skipIfNotPy311
    def test_bytecode_from_template_noprefix(self):
        # 测试3.11+版本的前缀指令是否被移除
        def gen_fn():
            cl = None

            def fn():
                return cl

            return fn

        # 生成函数fn
        fn = gen_fn()
        # 获取fn函数的字节码指令列表
        dis_insts = list(dis.get_instructions(fn))
        # 获取指令操作名称集合
        names = {inst.opname for inst in dis_insts}
        # 确保集合中包含"RESUME"
        self.assertIn("RESUME", names)
        # 确保集合中包含"COPY_FREE_VARS"
        self.assertIn("COPY_FREE_VARS", names)

        # 使用模板生成字节码指令列表
        insts = bytecode_transformation.bytecode_from_template(fn)
        # 获取指令操作名称集合
        names = {inst.opname for inst in insts}
        # 确保集合中不包含"RESUME"
        self.assertNotIn("RESUME", names)
        # 确保集合中不包含"COPY_FREE_VARS"
        self.assertNotIn("COPY_FREE_VARS", names)
    # 定义测试函数：验证带有多个返回语句的函数是否被替换为跳转到结尾的指令
    def test_bytecode_from_template_noreturn1(self):
        # 内部函数 fn，根据条件 x 返回 y 或返回 z
        def fn():
            if x:
                return y
            z = 3
            return z
        
        # 使用 dis 模块获取函数 fn 的指令列表
        dis_insts = list(dis.get_instructions(fn))
        # 筛选所有以 "RETURN" 开头的指令
        dis_returns = list(filter(lambda x: x.opname.startswith("RETURN"), dis_insts))
        # 断言有多于一个 RETURN 指令
        self.assertGreater(len(dis_returns), 1)
        # 断言最后一个指令是 "RETURN"
        self.assertTrue(dis_insts[-1].opname.startswith("RETURN"))

        # 对函数 fn 应用字节码转换，获取转换后的指令列表 insts
        insts = bytecode_transformation.bytecode_from_template(fn, noprefix=False)
        # 断言转换后的最后一个指令是 "NOP"
        self.assertEqual(insts[-1].opname, "NOP")
        # 断言指令列表长度与原始函数的指令列表长度相等
        self.assertEqual(len(dis_insts), len(insts))
        # 遍历原始指令列表 dis_insts 和转换后的指令列表 insts，检查 RETURN 替换为 JUMP
        for i0, i1 in zip(dis_insts, insts):
            if i0.opname.startswith("RETURN"):
                if i1 is insts[-1]:
                    continue
                # 断言转换后的指令包含 "JUMP" 并且跳转目标是最后一个指令
                self.assertIn("JUMP", i1.opname)
                self.assertIs(i1.target, insts[-1])

    # 应当在 Python 3.11+ 中正常工作，对 3.10 进行测试足够
    # 在 3.8 中，`fn` 结尾使用 RETURN_VALUE
    @skipIfNotPy311
    # 定义测试函数：验证不以 RETURN_VALUE 结尾的函数
    def test_bytecode_from_template_noreturn2(self):
        # 内部函数 fn，根据条件 x 返回 x 或抛出 RuntimeError
        def fn():
            if x:
                return x
            if x:
                return x
            raise RuntimeError
        
        # 使用 dis 模块获取函数 fn 的指令列表
        dis_insts = list(dis.get_instructions(fn))
        # 断言最后一个指令不是 "RETURN"
        self.assertFalse(dis_insts[-1].opname.startswith("RETURN"))

        # 对函数 fn 应用字节码转换，获取转换后的指令列表 insts
        insts = bytecode_transformation.bytecode_from_template(fn, noprefix=False)
        # 断言转换后的倒数第二个指令是和原始函数最后一个指令相同的操作码
        self.assertEqual(insts[-2].opname, dis_insts[-1].opname)
        # 断言转换后的指令列表长度比原始函数的指令列表长度多一
        self.assertEqual(len(dis_insts) + 1, len(insts))
        # 遍历原始指令列表 dis_insts 和转换后的指令列表 insts，检查 RETURN 替换为 JUMP
        for i0, i1 in zip(dis_insts, insts):
            if i0.opname.startswith("RETURN"):
                # 断言转换后的指令包含 "JUMP" 并且跳转目标是最后一个指令
                self.assertIn("JUMP", i1.opname)
                self.assertIs(i1.target, insts[-1])

    @skipIfNotPy312
    # 定义测试函数：验证 Python 3.12+ 中的 RETURN_CONST
    def test_bytecode_from_template_noreturn_const(self):
        # 内部函数 fn，根据条件 x 返回 1 或返回 0
        def fn():
            if x:
                return 1
            return 0
        
        # 使用 dis 模块获取函数 fn 的指令列表
        dis_insts = list(dis.get_instructions(fn))
        # 筛选所有操作码为 "RETURN_CONST" 的指令
        dis_return_consts = list(filter(lambda x: x.opname == "RETURN_CONST", dis_insts))
        # 断言有多于一个 RETURN_CONST 指令
        self.assertGreater(len(dis_return_consts), 1)
        # 断言最后一个指令是 "RETURN_CONST"
        self.assertTrue(dis_insts[-1].opname == "RETURN_CONST")

        # 对函数 fn 应用字节码转换，获取转换后的指令列表 insts
        insts = bytecode_transformation.bytecode_from_template(fn, noprefix=False)
        # 断言转换后的最后一个指令是 "NOP"
        self.assertEqual(insts[-1].opname, "NOP")
        insts_i = 0
        # 遍历原始指令列表 dis_insts，检查 RETURN_CONST 被转换为 LOAD_CONST 和 JUMP
        for i, inst in enumerate(dis_insts):
            if inst.opname == "RETURN_CONST":
                # 断言转换后的指令为 "LOAD_CONST"
                self.assertEqual(insts[insts_i].opname, "LOAD_CONST")
                insts_i += 1
                if insts_i != len(insts) - 1:
                    # 断言转换后的指令包含 "JUMP" 并且跳转目标是最后一个指令
                    self.assertIn("JUMP", insts[insts_i].opname)
                    self.assertIs(insts[insts_i].target, insts[-1])
            insts_i += 1
class BytecodeHookTests(torch._dynamo.test_case.TestCase):
    # 定义测试类 BytecodeHookTests，继承自 torch._dynamo.test_case.TestCase

    def test_bytecode_hook(self):
        # 定义测试方法 test_bytecode_hook

        def fn(a, b):
            # 定义函数 fn，接收参数 a 和 b，返回 a - b * 10 的结果
            return a - b * 10

        def hook(code, out_code):
            # 定义钩子函数 hook，接收两个参数 code 和 out_code
            # 打印输入的 code 和 out_code
            print(code)
            print(out_code)
            # 返回未修改的 code
            return code

        # 重置 torch._dynamo 的状态
        torch._dynamo.reset()
        
        # 注册 bytecode 钩子函数 hook 到 convert_frame 上，并保存返回的句柄
        handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
        
        try:
            # 编译函数 fn，返回优化后的函数 opt_fn
            opt_fn = torch.compile(fn)
            
            # 循环调用 opt_fn，传入不同维度的随机张量
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        finally:
            # 移除注册的 bytecode 钩子函数 handle
            handle.remove()

if __name__ == "__main__":
    # 如果当前脚本作为主程序执行

    # 导入 run_tests 函数从 torch._dynamo.test_case 模块中
    from torch._dynamo.test_case import run_tests

    # 运行测试用例
    run_tests()
```