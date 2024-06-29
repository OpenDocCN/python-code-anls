# `D:\src\scipysrc\pandas\doc\make.py`

```
#!/usr/bin/env python3
"""
Python script for building documentation.

To build the docs you must have all optional dependencies for pandas
installed. See the installation instructions for a list of these.

Usage
-----
    $ python make.py clean
    $ python make.py html
    $ python make.py latex
"""

# 引入必要的模块和库
import argparse                  # 解析命令行参数的库
import csv                       # 读写 CSV 文件的库
import importlib                 # 动态导入模块的库
import os                        # 提供了多种操作系统接口的库
import shutil                    # 文件操作相关的高级操作库
import subprocess                # 启动和管理子进程的库
import sys                       # 提供对 Python 解释器的访问
import webbrowser                # 启动一个新的浏览器进程打开 URL

import docutils                  # 文档处理工具
import docutils.parsers.rst       # reStructuredText 解析器

DOC_PATH = os.path.dirname(os.path.abspath(__file__))    # 获取当前脚本所在目录的绝对路径
SOURCE_PATH = os.path.join(DOC_PATH, "source")            # 源文件目录的路径
BUILD_PATH = os.path.join(DOC_PATH, "build")              # 构建结果存放目录的路径
REDIRECTS_FILE = os.path.join(DOC_PATH, "redirects.csv")  # 重定向文件的路径

class DocBuilder:
    """
    Class to wrap the different commands of this script.

    All public methods of this class can be called as parameters of the
    script.
    """

    def __init__(
        self,
        num_jobs="auto",           # 并行工作数，默认为自动选择
        include_api=True,          # 是否包含 API 文档，默认为是
        whatsnew=False,            # 是否只生成“更新内容”文档，默认为否
        single_doc=None,           # 单个文档的文件名
        verbosity=0,               # 输出详细级别，默认为0
        warnings_are_errors=False, # 将警告视为错误，默认为否
        no_browser=False,          # 是否禁用浏览器打开文档，默认为否
    ) -> None:
        self.num_jobs = num_jobs
        self.include_api = include_api
        self.whatsnew = whatsnew
        self.verbosity = verbosity
        self.warnings_are_errors = warnings_are_errors
        self.no_browser = no_browser

        # 设置环境变量 SPHINX_PATTERN，用于控制文档生成的模式
        if single_doc:
            single_doc = self._process_single_doc(single_doc)
            os.environ["SPHINX_PATTERN"] = single_doc
        elif not include_api:
            os.environ["SPHINX_PATTERN"] = "-api"
        elif whatsnew:
            os.environ["SPHINX_PATTERN"] = "whatsnew"

        self.single_doc_html = None
        # 如果指定了单个文档且其为 .rst 格式，则生成对应的 HTML 文件名
        if single_doc and single_doc.endswith(".rst"):
            self.single_doc_html = os.path.splitext(single_doc)[0] + ".html"
        elif single_doc:
            self.single_doc_html = f"reference/api/pandas.{single_doc}.html"
    def _process_single_doc(self, single_doc):
        """
        Make sure the provided value for --single is a path to an existing
        .rst/.ipynb file, or a pandas object that can be imported.

        For example, categorial.rst or pandas.DataFrame.head. For the latter,
        return the corresponding file path
        (e.g. reference/api/pandas.DataFrame.head.rst).
        """
        # Split the provided single_doc into base_name and extension
        base_name, extension = os.path.splitext(single_doc)
        
        # Check if the extension is .rst or .ipynb
        if extension in (".rst", ".ipynb"):
            # Construct the full path and check if it exists
            if os.path.exists(os.path.join(SOURCE_PATH, single_doc)):
                return single_doc  # Return the valid file path
            else:
                raise FileNotFoundError(f"File {single_doc} not found")

        # If single_doc starts with "pandas."
        elif single_doc.startswith("pandas."):
            try:
                obj = pandas  # noqa: F821
                # Traverse through the attributes of pandas object
                for name in single_doc.split("."):
                    obj = getattr(obj, name)
            except AttributeError as err:
                raise ImportError(f"Could not import {single_doc}") from err
            else:
                return single_doc[len("pandas.") :]  # Return the remaining attribute path
        else:
            # Raise error if the format of single_doc is not recognized
            raise ValueError(
                f"--single={single_doc} not understood. "
                "Value should be a valid path to a .rst or .ipynb file, "
                "or a valid pandas object "
                "(e.g. categorical.rst or pandas.DataFrame.head)"
            )

    @staticmethod
    def _run_os(*args) -> None:
        """
        Execute a command as a OS terminal.

        Parameters
        ----------
        *args : list of str
            Command and parameters to be executed

        Examples
        --------
        >>> DocBuilder()._run_os("python", "--version")
        """
        # Execute the given command in the OS terminal
        subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stderr)

    def _sphinx_build(self, kind: str):
        """
        Call sphinx to build documentation.

        Attribute `num_jobs` from the class is used.

        Parameters
        ----------
        kind : {'html', 'latex', 'linkcheck'}

        Examples
        --------
        >>> DocBuilder(num_jobs=4)._sphinx_build("html")
        """
        # Check if kind is one of the supported types
        if kind not in ("html", "latex", "linkcheck"):
            raise ValueError(f"kind must be html, latex or linkcheck, not {kind}")

        # Construct the sphinx-build command
        cmd = ["sphinx-build", "-b", kind]
        if self.num_jobs:
            cmd += ["-j", self.num_jobs]  # Add number of jobs if specified
        if self.warnings_are_errors:
            cmd += ["-W", "--keep-going"]  # Treat warnings as errors
        if self.verbosity:
            cmd.append(f"-{'v' * self.verbosity}")  # Add verbosity level
        cmd += [
            "-d",
            os.path.join(BUILD_PATH, "doctrees"),  # Path to doctrees directory
            SOURCE_PATH,  # Source path for documentation
            os.path.join(BUILD_PATH, kind),  # Destination path for built documentation
        ]
        return subprocess.call(cmd)  # Execute the sphinx-build command

    def _open_browser(self, single_doc_html) -> None:
        """
        Open a browser tab showing single

        Parameters
        ----------
        single_doc_html : str
            The HTML file path to open in the browser
        """
        # Construct the URL using the single_doc_html path
        url = os.path.join("file://", DOC_PATH, "build", "html", single_doc_html)
        # Open the URL in a new browser tab
        webbrowser.open(url, new=2)
        def _get_page_title(self, page):
            """
            Open the rst file `page` and extract its title.
            """
            # 构建指定页面的完整路径
            fname = os.path.join(SOURCE_PATH, f"{page}.rst")
            # 创建一个新的文档对象
            doc = docutils.utils.new_document(
                "<doc>",
                docutils.frontend.get_default_settings(docutils.parsers.rst.Parser),
            )
            # 使用 UTF-8 编码打开指定文件名的文件
            with open(fname, encoding="utf-8") as f:
                data = f.read()

            # 创建一个 reStructuredText 解析器对象
            parser = docutils.parsers.rst.Parser()
            # 在解析 rst 文件时，不生成任何警告
            with open(os.devnull, "a", encoding="utf-8") as f:
                doc.reporter.stream = f
                parser.parse(data, doc)

            # 找到文档中的第一个 section 节点
            section = next(
                node for node in doc.children if isinstance(node, docutils.nodes.section)
            )
            # 找到 section 节点中的 title 节点
            title = next(
                node for node in section.children if isinstance(node, docutils.nodes.title)
            )

            # 返回标题文本内容
            return title.astext()

        def _add_redirects(self) -> None:
            """
            Create in the build directory an html file with a redirect,
            for every row in REDIRECTS_FILE.
            """
            # 使用 UTF-8 编码打开重定向文件
            with open(REDIRECTS_FILE, encoding="utf-8") as mapping_fd:
                # 创建 CSV 读取器对象
                reader = csv.reader(mapping_fd)
                # 遍历 CSV 文件中的每一行
                for row in reader:
                    # 跳过空行和以 "#" 开头的注释行
                    if not row or row[0].strip().startswith("#"):
                        continue

                    # 构建 HTML 文件路径
                    html_path = os.path.join(BUILD_PATH, "html")
                    path = os.path.join(html_path, *row[0].split("/")) + ".html"

                    # 如果不包含 API 且路径包含 "reference" 或 "generated"，则跳过
                    if not self.include_api and (
                        os.path.join(html_path, "reference") in path
                        or os.path.join(html_path, "generated") in path
                    ):
                        continue

                    try:
                        # 尝试获取页面标题
                        title = self._get_page_title(row[1])
                    except Exception:
                        # 处理异常情况，例如文件可能是 ipynb 而不是 rst，或者 docutils
                        # 无法读取 rst 文件
                        title = "this page"

                    # 打开目标路径的文件，并写入重定向 HTML 内容
                    with open(path, "w", encoding="utf-8") as moved_page_fd:
                        html = f"""\
<html>
    <head>
        <meta http-equiv="refresh" content="0;URL={row[1]}.html"/>
        # 设置自动跳转到指定URL的页面
    </head>
    <body>
        <p>
            The page has been moved to <a href="{row[1]}.html">{title}</a>
            # 显示已经移动到新页面的提示信息，并提供链接到新页面
        </p>
    </body>
<html>"""

                    moved_page_fd.write(html)
                    # 将生成的HTML内容写入文件

    def html(self):
        """
        Build HTML documentation.
        """
        ret_code = self._sphinx_build("html")
        # 使用Sphinx构建HTML文档
        zip_fname = os.path.join(BUILD_PATH, "html", "pandas.zip")
        # 构建ZIP文件名路径

        if os.path.exists(zip_fname):
            os.remove(zip_fname)  # noqa: TID251
            # 如果ZIP文件存在，则删除它

        if ret_code == 0:
            if self.single_doc_html is not None:
                if not self.no_browser:
                    self._open_browser(self.single_doc_html)
                    # 如果指定了单个HTML文档，并且不禁用浏览器，则打开浏览器查看文档
            else:
                self._add_redirects()
                # 添加重定向页面
                if self.whatsnew and not self.no_browser:
                    self._open_browser(os.path.join("whatsnew", "index.html"))
                    # 如果启用了what's new页面，并且不禁用浏览器，则打开浏览器查看what's new页面

        return ret_code
        # 返回构建HTML文档的返回码

    def latex(self, force=False):
        """
        Build PDF documentation.
        """
        if sys.platform == "win32":
            sys.stderr.write("latex build has not been tested on windows\n")
            # 如果运行平台是Windows，则输出未在Windows上测试的提示信息
        else:
            ret_code = self._sphinx_build("latex")
            # 使用Sphinx构建LaTeX文档
            os.chdir(os.path.join(BUILD_PATH, "latex"))
            # 切换工作目录到LaTeX输出路径
            if force:
                for i in range(3):
                    self._run_os("pdflatex", "-interaction=nonstopmode", "pandas.tex")
                    # 如果强制构建，则使用pdflatex命令尝试构建PDF文档三次
                raise SystemExit(
                    "You should check the file "
                    '"build/latex/pandas.pdf" for problems.'
                )
                # 提示用户检查PDF文档是否存在问题
            self._run_os("make")
            # 使用make命令继续构建PDF文档
            return ret_code
            # 返回构建LaTeX文档的返回码

    def latex_forced(self):
        """
        Build PDF documentation with retries to find missing references.
        """
        return self.latex(force=True)
        # 使用强制模式构建PDF文档，并返回构建结果

    @staticmethod
    def clean() -> None:
        """
        Clean documentation generated files.
        """
        shutil.rmtree(BUILD_PATH, ignore_errors=True)
        # 递归删除生成的文档文件夹及其内容，如果存在错误则忽略

        shutil.rmtree(os.path.join(SOURCE_PATH, "reference", "api"), ignore_errors=True)
        # 递归删除API文档文件夹及其内容，如果存在错误则忽略

    def zip_html(self) -> None:
        """
        Compress HTML documentation into a zip file.
        """
        zip_fname = os.path.join(BUILD_PATH, "html", "pandas.zip")
        # 构建ZIP文件名路径

        if os.path.exists(zip_fname):
            os.remove(zip_fname)  # noqa: TID251
            # 如果ZIP文件存在，则删除它

        dirname = os.path.join(BUILD_PATH, "html")
        # 构建HTML文档路径

        fnames = os.listdir(dirname)
        # 获取HTML文档文件夹中的文件名列表

        os.chdir(dirname)
        # 切换工作目录到HTML文档路径

        self._run_os("zip", zip_fname, "-r", "-q", *fnames)
        # 使用zip命令将HTML文档文件夹中的所有文件压缩成一个ZIP文件

    def linkcheck(self):
        """
        Check for broken links in the documentation.
        """
        return self._sphinx_build("linkcheck")
        # 使用Sphinx检查文档中的损坏链接

def main():
    cmds = [method for method in dir(DocBuilder) if not method.startswith("_")]
    # 获取DocBuilder类中非私有方法的名称列表

    joined = ",".join(cmds)
    # 将方法名列表转换为逗号分隔的字符串

    argparser = argparse.ArgumentParser(
        description="pandas documentation builder", epilog=f"Commands: {joined}"
    )
    # 创建命令行参数解析器，并设置描述和结尾信息

    joined = ", ".join(cmds)
    # 将方法名列表转换为逗号分隔的字符串
    # 添加一个位置参数 'command'，默认为"html"，如果提供的话，可以是其他命令，例如'singlehtml'。
    argparser.add_argument(
        "command", nargs="?", default="html", help=f"command to run: {joined}"
    )
    # 添加一个可选参数 '--num-jobs'，默认为"auto"，用于指定sphinx-build使用的作业数量。
    argparser.add_argument(
        "--num-jobs", default="auto", help="number of jobs used by sphinx-build"
    )
    # 添加一个标志参数 '--no-api'，默认为False，如果设置为True，则不包含api和autosummary。
    argparser.add_argument(
        "--no-api", default=False, help="omit api and autosummary", action="store_true"
    )
    # 添加一个标志参数 '--whatsnew'，默认为False，如果设置为True，则仅构建whatsnew部分（以及用于链接的api）。
    argparser.add_argument(
        "--whatsnew",
        default=False,
        help="only build whatsnew (and api for links)",
        action="store_true",
    )
    # 添加一个参数 '--single'，用于指定要编译的文件名（相对于'source'文件夹），可以是章节或方法名。
    argparser.add_argument(
        "--single",
        metavar="FILENAME",
        type=str,
        default=None,
        help=(
            "filename (relative to the 'source' folder) of section or method name to "
            "compile, e.g. 'development/contributing.rst', "
            "'pandas.DataFrame.join'"
        ),
    )
    # 添加一个参数 '--python-path'，用于指定Python模块的路径，这将影响Sphinx的编译过程。
    argparser.add_argument(
        "--python-path", type=str, default=os.path.dirname(DOC_PATH), help="path"
    )
    # 添加一个标志参数 '-v'，用于增加输出详细程度的级别，可以重复多次。
    argparser.add_argument(
        "-v",
        action="count",
        dest="verbosity",
        default=0,
        help=(
            "increase verbosity (can be repeated), "
            "passed to the sphinx build command"
        ),
    )
    # 添加一个标志参数 '--warnings-are-errors'或'-W'，如果设置为True，则在引发警告时失败。
    argparser.add_argument(
        "--warnings-are-errors",
        "-W",
        action="store_true",
        help="fail if warnings are raised",
    )
    # 添加一个标志参数 '--no-browser'，默认为False，如果设置为True，则不打开浏览器。
    argparser.add_argument(
        "--no-browser",
        help="Don't open browser",
        default=False,
        action="store_true",
    )
    # 解析命令行参数并返回结果。
    args = argparser.parse_args()

    # 如果提供的命令不在cmds列表中，则抛出ValueError异常。
    if args.command not in cmds:
        joined = ", ".join(cmds)
        raise ValueError(f"Unknown command {args.command}. Available options: {joined}")

    # 设置环境变量'PYTHONPATH'为指定的Python路径。
    os.environ["PYTHONPATH"] = args.python_path
    # 将args.python_path插入sys.path的开头，以便在模块中正确解析导入。
    sys.path.insert(0, args.python_path)
    # 动态导入pandas模块并将其添加到全局命名空间中。
    globals()["pandas"] = importlib.import_module("pandas")

    # 将matplotlib后端设置为非交互式的Agg后端，适用于所有子进程。
    os.environ["MPLBACKEND"] = "module://matplotlib.backends.backend_agg"

    # 创建DocBuilder对象并执行命令args.command，返回执行结果。
    builder = DocBuilder(
        args.num_jobs,
        not args.no_api,
        args.whatsnew,
        args.single,
        args.verbosity,
        args.warnings_are_errors,
        args.no_browser,
    )
    return getattr(builder, args.command)()
# 如果当前脚本被作为主程序执行（而不是被导入到其他模块），则执行以下代码块
if __name__ == "__main__":
    # 调用 main 函数，并终止程序，返回 main 函数的返回码作为退出码
    sys.exit(main())
```