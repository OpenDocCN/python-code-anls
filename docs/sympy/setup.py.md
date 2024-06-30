# `D:\src\scipysrc\sympy\setup.py`

```
#!/usr/bin/env python
"""Setup script for SymPy.

This uses Setuptools (https://setuptools.pypa.io/en/latest/) the standard
python mechanism for installing packages.
For the easiest installation just type the command (you'll probably need
root privileges for that):

    pip install .

This will install the library in the default location. For instructions on
how to customize the installation procedure read the output of:

    pip install --help

In addition, there are some other commands:

    python setup.py test  -> will run the complete test suite

To get a full list of available commands, read the output of:

    python setup.py --help-commands

Or, if all else fails, feel free to write to the sympy list at
sympy@googlegroups.com and ask for help.
"""

import sys  # 导入系统模块
import os  # 导入操作系统功能模块
import subprocess  # 导入子进程管理模块
from pathlib import Path  # 导入路径操作模块

from setuptools import setup, Command  # 导入setuptools中的setup和Command类
from setuptools.command.sdist import sdist  # 导入setuptools中的sdist类


# This directory
dir_setup = os.path.dirname(os.path.realpath(__file__))  # 获取当前脚本所在目录的路径

extra_kwargs = {
    'zip_safe': False,
    'entry_points': {
        'console_scripts': [
            'isympy = isympy:main',
        ]
    }
}

if sys.version_info < (3, 8):
    print("SymPy requires Python 3.8 or newer. Python %d.%d detected"
          % sys.version_info[:2])  # 检查Python版本是否符合要求，并输出警告信息
    sys.exit(-1)  # 如果Python版本不符合要求，退出安装过程并返回错误码-1

# Check that this list is uptodate against the result of the command:
# python bin/generate_module_list.py
modules = [
    'sympy.algebras',  # 列出需要包含的模块路径
    'sympy.assumptions',
    'sympy.assumptions.handlers',
    'sympy.assumptions.predicates',
    'sympy.assumptions.relation',
    'sympy.benchmarks',
    'sympy.calculus',
    'sympy.categories',
    'sympy.codegen',
    'sympy.combinatorics',
    'sympy.concrete',
    'sympy.core',
    'sympy.core.benchmarks',
    'sympy.crypto',
    'sympy.diffgeom',
    'sympy.discrete',
    'sympy.external',
    'sympy.functions',
    'sympy.functions.combinatorial',
    'sympy.functions.elementary',
    'sympy.functions.elementary.benchmarks',
    'sympy.functions.special',
    'sympy.functions.special.benchmarks',
    'sympy.geometry',
    'sympy.holonomic',
    'sympy.integrals',
    'sympy.integrals.benchmarks',
    'sympy.interactive',
    'sympy.liealgebras',
    'sympy.logic',
    'sympy.logic.algorithms',
    'sympy.logic.utilities',
    'sympy.matrices',
    'sympy.matrices.benchmarks',
    'sympy.matrices.expressions',
    'sympy.multipledispatch',
    'sympy.ntheory',
    'sympy.parsing',
    'sympy.parsing.autolev',
    'sympy.parsing.autolev._antlr',
    'sympy.parsing.c',
    'sympy.parsing.fortran',
    'sympy.parsing.latex',
    'sympy.parsing.latex._antlr',
    'sympy.parsing.latex.lark',
    'sympy.physics',
    'sympy.physics.biomechanics',
    'sympy.physics.continuum_mechanics',
    'sympy.physics.control',
    'sympy.physics.hep',
    'sympy.physics.mechanics',
    'sympy.physics.optics',
    'sympy.physics.quantum',
    'sympy.physics.units',
    'sympy.physics.units.definitions',
    'sympy.physics.units.systems',
    # 导入需要的 SymPy 模块
    import sympy.physics.vector
    import sympy.plotting
    import sympy.plotting.backends
    import sympy.plotting.backends.matplotlibbackend
    import sympy.plotting.backends.textbackend
    import sympy.plotting.intervalmath
    import sympy.plotting.pygletplot
    import sympy.polys
    import sympy.polys.agca
    import sympy.polys.benchmarks
    import sympy.polys.domains
    import sympy.polys.matrices
    import sympy.polys.numberfields
    import sympy.printing
    import sympy.printing.pretty
    import sympy.sandbox
    import sympy.series
    import sympy.series.benchmarks
    import sympy.sets
    import sympy.sets.handlers
    import sympy.simplify
    import sympy.solvers
    import sympy.solvers.benchmarks
    import sympy.solvers.diophantine
    import sympy.solvers.ode
    import sympy.stats
    import sympy.stats.sampling
    import sympy.strategies
    import sympy.strategies.branch
    import sympy.tensor
    import sympy.tensor.array
    import sympy.tensor.array.expressions
    import sympy.testing
    import sympy.unify
    import sympy.utilities
    import sympy.utilities._compilation
    import sympy.utilities.mathml
    import sympy.utilities.mathml.data
    import sympy.vector
# 定义一个名为 test_sympy 的类，继承自 Command 类
class test_sympy(Command):
    """Runs all tests under the sympy/ folder
    """

    # 描述信息：运行 sympy/ 文件夹下所有测试
    description = "run all tests and doctests; also see bin/test and bin/doctest"
    user_options = []  # setuptools 如果没有这个会报错。

    # 初始化方法，接收所有参数并存储在 self.args 中，以便传递给其他类
    def __init__(self, *args):
        self.args = args[0]  # 为了能够将其传递给其他类
        Command.__init__(self, *args)

    # setuptools 要求的方法，用于初始化选项
    def initialize_options(self):
        pass

    # setuptools 要求的方法，用于完成选项初始化
    def finalize_options(self):
        pass

    # 实际运行方法，导入 sympy.testing 中的 runtests 模块，并运行所有测试
    def run(self):
        from sympy.testing import runtests
        runtests.run_all_tests()


# 定义一个名为 antlr 的类，用于生成 antlr4 的代码
class antlr(Command):
    """Generate code with antlr4"""
    # 描述信息：使用 antlr4 生成解析器代码
    description = "generate parser code from antlr grammars"
    user_options = []  # setuptools 如果没有这个会报错。

    # 初始化方法，接收所有参数并存储在 self.args 中，以便传递给其他类
    def __init__(self, *args):
        self.args = args[0]  # 为了能够将其传递给其他类
        Command.__init__(self, *args)

    # setuptools 要求的方法，用于初始化选项
    def initialize_options(self):
        pass

    # setuptools 要求的方法，用于完成选项初始化
    def finalize_options(self):
        pass

    # 实际运行方法，从 sympy.parsing.latex._build_latex_antlr 中导入 build_parser 函数，
    # 尝试构建 LaTeX 解析器，如果失败则退出程序
    def run(self):
        from sympy.parsing.latex._build_latex_antlr import build_parser as build_latex_parser
        if not build_latex_parser():
            sys.exit(-1)

        # 从 sympy.parsing.autolev._build_autolev_antlr 中导入 build_parser 函数，
        # 尝试构建 Autolev 解析器，如果失败则退出程序
        from sympy.parsing.autolev._build_autolev_antlr import build_parser as build_autolev_parser
        if not build_autolev_parser():
            sys.exit(-1)


# 定义一个名为 sdist_sympy 的类，继承自 sdist 类
class sdist_sympy(sdist):
    # 运行方法，重写 sdist 类的 run 方法
    def run(self):
        # 在打包前获取 Git 提交哈希，并写入 commit_hash.txt 文件
        commit_hash = None
        commit_hash_filepath = 'doc/commit_hash.txt'
        try:
            commit_hash = \
                subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            commit_hash = commit_hash.decode('ascii')
            commit_hash = commit_hash.rstrip()
            print('Commit hash found : {}.'.format(commit_hash))
            print('Writing it to {}.'.format(commit_hash_filepath))
        except:
            pass

        # 如果成功获取到 commit_hash，则将其写入 commit_hash_filepath 文件中
        if commit_hash:
            with open(commit_hash_filepath, 'w') as f:
                f.write(commit_hash)

        # 调用父类 sdist 的 run 方法执行打包过程
        super().run()

        # 尝试删除 commit_hash_filepath 文件，打印成功删除的信息；如果失败则打印异常信息
        try:
            os.remove(commit_hash_filepath)
            print(
                'Successfully removed temporary file {}.'
                .format(commit_hash_filepath))
        except OSError as e:
            print("Error deleting %s - %s." % (e.filename, e.strerror))


# 检查这个列表是否与命令 python bin/generate_test_list.py 的输出结果保持一致
tests = [
    'sympy.algebras.tests',
    'sympy.assumptions.tests',
    'sympy.calculus.tests',
    'sympy.categories.tests',
    'sympy.codegen.tests',
    'sympy.combinatorics.tests',
    'sympy.concrete.tests',
    'sympy.core.tests',
    'sympy.crypto.tests',
    'sympy.diffgeom.tests',
    'sympy.discrete.tests',
    'sympy.external.tests',
    'sympy.functions.combinatorial.tests',
    'sympy.functions.elementary.tests',
]
    # 创建一个包含多个字符串的列表，每个字符串代表一个测试模块的名称
    modules = [
        'sympy.functions.special.tests',
        'sympy.geometry.tests',
        'sympy.holonomic.tests',
        'sympy.integrals.tests',
        'sympy.interactive.tests',
        'sympy.liealgebras.tests',
        'sympy.logic.tests',
        'sympy.matrices.expressions.tests',
        'sympy.matrices.tests',
        'sympy.multipledispatch.tests',
        'sympy.ntheory.tests',
        'sympy.parsing.tests',
        'sympy.physics.biomechanics.tests',
        'sympy.physics.continuum_mechanics.tests',
        'sympy.physics.control.tests',
        'sympy.physics.hep.tests',
        'sympy.physics.mechanics.tests',
        'sympy.physics.optics.tests',
        'sympy.physics.quantum.tests',
        'sympy.physics.tests',
        'sympy.physics.units.tests',
        'sympy.physics.vector.tests',
        'sympy.plotting.intervalmath.tests',
        'sympy.plotting.pygletplot.tests',
        'sympy.plotting.tests',
        'sympy.polys.agca.tests',
        'sympy.polys.domains.tests',
        'sympy.polys.matrices.tests',
        'sympy.polys.numberfields.tests',
        'sympy.polys.tests',
        'sympy.printing.pretty.tests',
        'sympy.printing.tests',
        'sympy.sandbox.tests',
        'sympy.series.tests',
        'sympy.sets.tests',
        'sympy.simplify.tests',
        'sympy.solvers.diophantine.tests',
        'sympy.solvers.ode.tests',
        'sympy.solvers.tests',
        'sympy.stats.sampling.tests',
        'sympy.stats.tests',
        'sympy.strategies.branch.tests',
        'sympy.strategies.tests',
        'sympy.tensor.array.expressions.tests',
        'sympy.tensor.array.tests',
        'sympy.tensor.tests',
        'sympy.testing.tests',
        'sympy.unify.tests',
        'sympy.utilities._compilation.tests',
        'sympy.utilities.tests',
        'sympy.vector.tests',
    ]
# 使用 open 函数打开指定路径下的文件 'sympy/release.py'，并将其内容读取为字符串，然后执行这段代码
with open(os.path.join(dir_setup, 'sympy', 'release.py')) as f:
    exec(f.read())

# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == '__main__':
    # 设置安装参数和元数据，用于安装和发布 SymPy 库
    setup(name='sympy',
          version=__version__,  # 使用之前从 release.py 中定义的 __version__ 变量作为版本号
          description='Computer algebra system (CAS) in Python',  # 简短描述
          long_description=(Path(__file__).parent / 'README.md').read_text("UTF-8"),  # 从 README.md 文件读取长描述
          long_description_content_type='text/markdown',  # 长描述的内容类型为 Markdown
          author='SymPy development team',  # 作者信息
          author_email='sympy@googlegroups.com',  # 作者邮箱
          license='BSD',  # 许可证类型
          keywords="Math CAS",  # 关键字
          url='https://sympy.org',  # 项目主页
          project_urls={
              'Source': 'https://github.com/sympy/sympy',  # 其他项目相关链接
          },
          # 设置安装时的依赖项，需要 mpmath 版本 >= 1.1.0
          install_requires=[
              'mpmath >= 1.1.0',
          ],
          py_modules=['isympy'],  # 需要安装的单文件模块
          packages=['sympy'] + modules + tests,  # 需要安装的包列表，包括 sympy 以及动态确定的 modules 和 tests
          ext_modules=[],  # 扩展模块列表为空
          # 配置各个包的附加数据文件路径
          package_data={
              'sympy.utilities.mathml.data': ['*.xsl'],
              'sympy.logic.benchmarks': ['input/*.cnf'],
              'sympy.parsing.autolev': [
                  '*.g4', 'test-examples/*.al', 'test-examples/*.py',
                  'test-examples/pydy-example-repo/*.al',
                  'test-examples/pydy-example-repo/*.py',
                  'test-examples/README.txt',
              ],
              'sympy.parsing.latex': ['*.txt', '*.g4', 'lark/grammar/*.lark'],
              'sympy.plotting.tests': ['test_region_*.png'],
              'sympy': ['py.typed']
          },
          # 安装的额外数据文件，如 man 页面
          data_files=[('share/man/man1', ['doc/man/isympy.1'])],
          # 自定义命令行工具的类
          cmdclass={'test': test_sympy,
                    'antlr': antlr,
                    'sdist': sdist_sympy,
                    },
          python_requires='>=3.8',  # 要求 Python 版本 >= 3.8
          # 分类器列表，描述该软件包的性质
          classifiers=[
              'License :: OSI Approved :: BSD License',
              'Operating System :: OS Independent',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Physics',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
              'Programming Language :: Python :: 3.10',
              'Programming Language :: Python :: 3.11',
              'Programming Language :: Python :: 3 :: Only',
              'Programming Language :: Python :: Implementation :: CPython',
              'Programming Language :: Python :: Implementation :: PyPy',
          ],
          extras_require={
              "dev": ["pytest>=7.1.0", "hypothesis>=6.70.0"],  # 额外的开发依赖项
          },
          **extra_kwargs  # 其他的关键字参数，用于覆盖默认设置
          )
```