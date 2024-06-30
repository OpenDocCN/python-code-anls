# `D:\src\scipysrc\sympy\release\compare_tar_against_git.py`

```
#!/usr/bin/env python3

# 导入子进程模块中的check_output函数
from subprocess import check_output
# 导入sys模块，用于处理系统相关的功能
import sys
# 导入os.path模块，用于处理路径相关的功能
import os.path


# 主函数定义，接受两个参数：tarname和gitroot
def main(tarname, gitroot):
    """Run this as ./compare_tar_against_git.py TARFILE GITROOT

    Args
    ====

    TARFILE: Path to the built sdist (sympy-xx.tar.gz)
    GITROOT: Path ro root of git (dir containing .git)
    """
    # 调用比较函数，比较打包文件和Git仓库的内容
    compare_tar_against_git(tarname, gitroot)


## TARBALL WHITELISTS

# 如果某个文件未包含在tarball中但应该包含，需将其添加到setup.py（如果是Python文件）或MANIFEST.in（如果不是Python文件）中。
# （setup.py文件开头有一条命令用于收集所有应包含的内容）

# TODO: 还需检查白名单是否未包含自Git中已删除的文件。

# 在Git中存在但不应包含在tarball中的文件列表
git_whitelist = {
    # Git特定的点文件
    '.gitattributes',
    '.gitignore',
    '.mailmap',
    # CI
    '.github/PULL_REQUEST_TEMPLATE.md',
    '.github/workflows/runtests.yml',
    '.github/workflows/ci-sage.yml',
    '.github/workflows/comment-on-pr.yml',
    '.github/workflows/release.yml',
    '.github/workflows/docs-preview.yml',
    '.github/workflows/checkconflict.yml',
    '.ci/durations.json',
    '.ci/generate_durations_log.sh',
    '.ci/parse_durations_log.py',
    '.ci/blacklisted.json',
    '.ci/README.rst',
    '.circleci/config.yml',
    '.github/FUNDING.yml',
    '.editorconfig',
    '.coveragerc',
    '.flake8',
    'CODEOWNERS',
    'asv.conf.actions.json',
    'codecov.yml',
    'requirements-dev.txt',
    'MANIFEST.in',
    'banner.svg',
    # 行为准则
    'CODE_OF_CONDUCT.md',
    # 贡献指南
    'CONTRIBUTING.md',
    # 引用配置
    'CITATION.cff',
    # bin/目录中的内容，除非有意安装，否则不应包含在发行版中。
    # 大部分为开发用途。在tarball中运行测试，请使用setup.py test，或导入sympy并运行sympy.test()或sympy.doctest()。
    'bin/adapt_paths.py',
    'bin/ask_update.py',
    'bin/authors_update.py',
    'bin/build_doc.sh',
    'bin/coverage_doctest.py',
    'bin/coverage_report.py',
    'bin/deploy_doc.sh',
    'bin/diagnose_imports',
    'bin/doctest',
    'bin/generate_module_list.py',
    'bin/generate_test_list.py',
    'bin/get_sympy.py',
    'bin/mailmap_update.py',
    'bin/py.bench',
    'bin/strip_whitespace',
    'bin/sympy_time.py',
    'bin/sympy_time_cache.py',
    'bin/test',
    'bin/test_external_imports.py',
    'bin/test_executable.py',
    'bin/test_import',
    'bin/test_import.py',
    'bin/test_isolated',
    'bin/test_py2_import.py',
    'bin/test_setup.py',
    'bin/test_submodule_imports.py',
    'bin/test_optional_dependencies.py',
    'bin/test_sphinx.sh',
    'bin/mailmap_check.py',
    'bin/test_symengine.py',
    'bin/test_tensorflow.py',
    'bin/test_pyodide.mjs',
    # 这些笔记本尚未准备好发布。需要进行清理和最好进行文档测试。
    # 参见https://github.com/sympy/sympy/issues/6039。
    # 定义一个包含文件路径的列表，这些路径指向示例和发布相关的文件
    files = [
        'examples/advanced/identitysearch_example.ipynb',
        'examples/beginner/plot_advanced.ipynb',
        'examples/beginner/plot_colors.ipynb',
        'examples/beginner/plot_discont.ipynb',
        'examples/beginner/plot_gallery.ipynb',
        'examples/beginner/plot_intro.ipynb',
        'examples/intermediate/limit_examples_advanced.ipynb',
        'examples/intermediate/schwarzschild.ipynb',
        'examples/notebooks/density.ipynb',
        'examples/notebooks/fidelity.ipynb',
        'examples/notebooks/fresnel_integrals.ipynb',
        'examples/notebooks/qubits.ipynb',
        'examples/notebooks/sho1d_example.ipynb',
        'examples/notebooks/spin.ipynb',
        'examples/notebooks/trace.ipynb',
        'examples/notebooks/Bezout_Dixon_resultant.ipynb',
        'examples/notebooks/IntegrationOverPolytopes.ipynb',
        'examples/notebooks/Macaulay_resultant.ipynb',
        'examples/notebooks/Sylvester_resultant.ipynb',
        'examples/notebooks/README.txt',
        # 示例和发布相关的一些脚本和文件
        'release/.gitignore',
        'release/README.md',
        'release/compare_tar_against_git.py',
        'release/update_docs.py',
        'release/build_docs.py',
        'release/github_release.py',
        'release/helpers.py',
        'release/releasecheck.py',
        'release/sha256.py',
        'release/authors.py',
        'release/ci_release_script.sh',
        # 与 pytest 相关的文件和依赖
        'conftest.py',
        'requirements-dev.txt',
    ]
# 需要在 tar 文件中的文件不应该出现在 git 中的白名单

tarball_whitelist = {
    # 由 setup.py 生成的，包含 PyPI 的元数据
    "PKG-INFO",
    # 由 setuptools 生成的，更多的元数据
    'setup.cfg',
    'sympy.egg-info/PKG-INFO',
    'sympy.egg-info/SOURCES.txt',
    'sympy.egg-info/dependency_links.txt',
    'sympy.egg-info/requires.txt',
    'sympy.egg-info/top_level.txt',
    'sympy.egg-info/not-zip-safe',
    'sympy.egg-info/entry_points.txt',
    # 不确定是从哪里生成的...
    'doc/commit_hash.txt',
    }


def blue(text):
    # 返回带有蓝色 ANSI 转义序列的文本
    return "\033[34m%s\033[0m" % text


def red(text):
    # 返回带有红色 ANSI 转义序列的文本
    return "\033[31m%s\033[0m" % text


def run(*cmdline, cwd=None):
    """
    在子进程中运行命令并获取输出的行列表
    """
    return check_output(cmdline, encoding='utf-8', cwd=cwd).splitlines()


def full_path_split(path):
    """
    对路径执行完整的分割操作
    """
    # 基于 https://stackoverflow.com/a/13505966/161801
    rest, tail = os.path.split(path)
    if not rest or rest == os.path.sep:
        return (tail,)
    return full_path_split(rest) + (tail,)


def compare_tar_against_git(tarname, gitroot):
    """
    比较 tar 文件与 git ls-files 的内容

    请查看文件底部的白名单。
    """
    git_lsfiles = {i.strip() for i in run('git', 'ls-files', cwd=gitroot)}
    tar_output_orig = set(run('tar', 'tf', tarname))
    tar_output = set()
    for file in tar_output_orig:
        # tar 文件类似于 sympy-0.7.3/sympy/__init__.py，而 git 文件类似于 sympy/__init__.py。
        split_path = full_path_split(file)
        if split_path[-1]:
            # 排除目录，因为 git ls-files 不包括它们
            tar_output.add(os.path.join(*split_path[1:]))
    # print tar_output
    # print git_lsfiles
    fail = False
    print()
    print(blue("在 tar 文件中来自 git 的不应该存在的文件:"))
    print()
    for line in sorted(tar_output.intersection(git_whitelist)):
        fail = True
        print(line)
    print()
    print(blue("在 git 中存在但不在 tar 文件中的文件:"))
    print()
    for line in sorted(git_lsfiles - tar_output - git_whitelist):
        fail = True
        print(line)
    print()
    print(blue("在 tar 文件中存在但不在 git 中的文件:"))
    print()
    for line in sorted(tar_output - git_lsfiles - tarball_whitelist):
        fail = True
        print(line)
    print()

    if fail:
        sys.exit(red("在 tar 文件中发现或未发现的非白名单文件"))

if __name__ == "__main__":
    main(*sys.argv[1:])
```