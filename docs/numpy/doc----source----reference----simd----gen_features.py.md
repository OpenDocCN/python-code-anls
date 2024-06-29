# `.\numpy\doc\source\reference\simd\gen_features.py`

```
"""
Generate CPU features tables from CCompilerOpt
"""
# 导入必要的模块和类
from os import sys, path
from numpy.distutils.ccompiler_opt import CCompilerOpt

class FakeCCompilerOpt(CCompilerOpt):
    # 禁用缓存，因为不需要
    conf_nocache = True

    def __init__(self, arch, cc, *args, **kwargs):
        # 初始化虚拟编译器信息
        self.fake_info = (arch, cc, '')
        # 调用父类的初始化方法
        CCompilerOpt.__init__(self, None, **kwargs)

    def dist_compile(self, sources, flags, **kwargs):
        # 编译源文件，简化返回所有源文件
        return sources

    def dist_info(self):
        # 返回虚拟编译器信息
        return self.fake_info

    @staticmethod
    def dist_log(*args, stderr=False):
        # 静态方法，用于记录日志，这里是避免打印
        pass

    def feature_test(self, name, force_flags=None, macros=[]):
        # 进行特性测试，假设总是返回 True，用于加速
        return True

class Features:
    def __init__(self, arch, cc):
        # 初始化特性对象，使用自定义的虚拟编译器选项
        self.copt = FakeCCompilerOpt(arch, cc, cpu_baseline="max")

    def names(self):
        # 返回CPU基线名称列表
        return self.copt.cpu_baseline_names()

    def serialize(self, features_names):
        # 序列化特性信息
        result = []
        for f in self.copt.feature_sorted(features_names):
            gather = self.copt.feature_supported.get(f, {}).get("group", [])
            implies = self.copt.feature_sorted(self.copt.feature_implies(f))
            result.append((f, implies, gather))
        return result

    def table(self, **kwargs):
        # 生成特性表格
        return self.gen_table(self.serialize(self.names()), **kwargs)

    def table_diff(self, vs, **kwargs):
        # 比较两个特性对象之间的差异
        fnames = set(self.names())
        fnames_vs = set(vs.names())
        common = fnames.intersection(fnames_vs)
        extra = fnames.difference(fnames_vs)
        notavl = fnames_vs.difference(fnames)
        iextra = {}
        inotavl = {}
        idiff = set()
        for f in common:
            implies = self.copt.feature_implies(f)
            implies_vs = vs.copt.feature_implies(f)
            e = implies.difference(implies_vs)
            i = implies_vs.difference(implies)
            if not i and not e:
                continue
            if e:
                iextra[f] = e
            if i:
                inotavl[f] = e
            idiff.add(f)

        def fbold(f):
            # 根据特性是否在extra或notavl集合中返回不同的格式
            if f in extra:
                return f':enabled:`{f}`'
            if f in notavl:
                return f':disabled:`{f}`'
            return f

        def fbold_implies(f, i):
            # 根据特性是否在iextra或inotavl集合中返回不同的格式
            if i in iextra.get(f, {}):
                return f':enabled:`{i}`'
            if f in notavl or i in inotavl.get(f, {}):
                return f':disabled:`{i}`'
            return i

        # 将所有差异特性序列化并生成特性表格
        diff_all = self.serialize(idiff.union(extra))
        diff_all += vs.serialize(notavl)
        content = self.gen_table(
            diff_all, fstyle=fbold, fstyle_implies=fbold_implies, **kwargs
        )
        return content
    # 生成一个表格的函数，接受多个参数，其中 serialized_features 是序列化特征的列表，
    # fstyle 是一个函数，默认为 lambda 表达式，用于格式化特征名字
    # fstyle_implies 也是一个函数，默认为 lambda 表达式，用于格式化特征暗示
    # **kwargs 是额外的关键字参数
    def gen_table(self, serialized_features, fstyle=None, fstyle_implies=None,
                  **kwargs):

        # 如果 fstyle 没有提供，使用默认的 lambda 函数格式化特征名字
        if fstyle is None:
            fstyle = lambda ft: f'``{ft}``'
        
        # 如果 fstyle_implies 没有提供，使用默认的 lambda 函数格式化特征暗示
        if fstyle_implies is None:
            fstyle_implies = lambda origin, ft: fstyle(ft)

        # 初始化空列表 rows 用于存储表格的每一行
        rows = []
        # 初始化标志 have_gather，用于标记是否存在 gather 类型的特征
        have_gather = False
        
        # 遍历 serialized_features 中的每个元素 (f, implies, gather)
        for f, implies, gather in serialized_features:
            # 如果 gather 为真值（非空），则设置 have_gather 为 True
            if gather:
                have_gather = True
            
            # 使用 fstyle 函数格式化特征名字 f
            name = fstyle(f)
            # 使用 fstyle_implies 函数格式化 implies 列表中的每个元素
            implies = ' '.join([fstyle_implies(f, i) for i in implies])
            # 使用 fstyle_implies 函数格式化 gather 列表中的每个元素
            gather = ' '.join([fstyle_implies(f, i) for i in gather])
            # 将格式化后的 (name, implies, gather) 添加到 rows 中
            rows.append((name, implies, gather))
        
        # 如果 rows 列表为空，则返回空字符串
        if not rows:
            return ''
        
        # 初始化字段列表 fields
        fields = ["Name", "Implies", "Gathers"]
        
        # 如果没有 gather 类型的特征，删除 fields 中的最后一个元素
        if not have_gather:
            del fields[2]
            # 更新 rows，只保留 name 和 implies 两列
            rows = [(name, implies) for name, implies, _ in rows]
        
        # 调用 gen_rst_table 方法生成并返回一个 reStructuredText 格式的表格
        return self.gen_rst_table(fields, rows, **kwargs)

    # 生成 reStructuredText 格式的表格，接受字段名和行数据
    def gen_rst_table(self, field_names, rows, tab_size=4):
        # 断言条件：如果 rows 为空或者 field_names 的长度等于 rows 的第一行的长度
        assert(not rows or len(field_names) == len(rows[0]))
        
        # 在 rows 中添加 field_names 作为表格的首行
        rows.append(field_names)
        
        # 计算字段的长度列表
        fld_len = len(field_names)
        cls_len = [max(len(c[i]) for c in rows) for i in range(fld_len)]
        
        # 根据字段长度列表生成表格的边框
        cformat = ' '.join('{:<%d}' % i for i in cls_len)
        border = cformat.format(*['='*i for i in cls_len])

        # 对每一行数据进行格式化
        rows = [cformat.format(*row) for row in rows]
        
        # 添加表格的头部和底部边框
        rows = [border, cformat.format(*field_names), border] + rows
        
        # 添加表格的底部边框
        rows += [border]
        
        # 在每一行数据前添加指定大小的左边距
        rows = [(' ' * tab_size) + r for r in rows]
        
        # 返回格式化后的表格内容，使用换行符连接每一行
        return '\n'.join(rows)
# 定义一个函数，生成包含标题和内容的文本段落，内容用表格格式化
def wrapper_section(title, content, tab_size=4):
    tab = ' '*tab_size
    # 如果内容不为空，则生成带标题的文本段落
    if content:
        return (
            f"{title}\n{'~'*len(title)}"  # 标题及其下方的波浪线
            f"\n.. table::\n{tab}:align: left\n\n"  # 开始定义表格
            f"{content}\n\n"  # 添加表格内容
        )
    return ''  # 内容为空时返回空字符串

# 定义一个函数，生成包含标题和表格的标签页
def wrapper_tab(title, table, tab_size=4):
    tab = ' '*tab_size
    # 如果表格不为空，则生成包含标题和表格的标签页
    if table:
        ('\n' + tab).join((
            '.. tab:: ' + title,  # 标签页标题
            tab + '.. table::',  # 定义表格
            tab + 'align: left',  # 设置表格对齐方式
            table + '\n\n'  # 添加表格内容
        ))
    return ''  # 表格为空时返回空字符串

# 主程序入口
if __name__ == '__main__':

    # 美化后的架构名称映射表
    pretty_names = {
        "PPC64": "IBM/POWER big-endian",
        "PPC64LE": "IBM/POWER little-endian",
        "S390X": "IBM/ZSYSTEM(S390X)",
        "ARMHF": "ARMv7/A32",
        "AARCH64": "ARMv8/A64",
        "ICC": "Intel Compiler",
        # "ICCW": "Intel Compiler msvc-like",
        "MSVC": "Microsoft Visual C/C++"
    }

    # 生成路径：当前脚本所在目录下的generated_tables文件夹
    gen_path = path.join(
        path.dirname(path.realpath(__file__)), "generated_tables"
    )

    # 打开并写入cpu_features.inc文件
    with open(path.join(gen_path, 'cpu_features.inc'), 'w') as fd:
        fd.write(f'.. generated via {__file__}\n\n')  # 写入生成信息
        # 遍历架构列表生成表格段落
        for arch in (
            ("x86", "PPC64", "PPC64LE", "ARMHF", "AARCH64", "S390X")
        ):
            title = "On " + pretty_names.get(arch, arch)  # 获取美化后的架构名称
            table = Features(arch, 'gcc').table()  # 调用Features类生成表格内容
            fd.write(wrapper_section(title, table))  # 调用wrapper_section生成段落并写入文件

    # 打开并写入compilers-diff.inc文件
    with open(path.join(gen_path, 'compilers-diff.inc'), 'w') as fd:
        fd.write(f'.. generated via {__file__}\n\n')  # 写入生成信息
        # 遍历架构和编译器对生成表格段落
        for arch, cc_names in (
            ("x86", ("clang", "ICC", "MSVC")),
            ("PPC64", ("clang",)),
            ("PPC64LE", ("clang",)),
            ("ARMHF", ("clang",)),
            ("AARCH64", ("clang",)),
            ("S390X", ("clang",))
        ):
            arch_pname = pretty_names.get(arch, arch)  # 获取美化后的架构名称
            # 遍历编译器列表生成表格段落
            for cc in cc_names:
                title = f"On {arch_pname}::{pretty_names.get(cc, cc)}"  # 构造标题
                # 调用Features类生成表格差异内容
                table = Features(arch, cc).table_diff(Features(arch, "gcc"))
                fd.write(wrapper_section(title, table))  # 调用wrapper_section生成段落并写入文件
```