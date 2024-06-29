# `D:\src\scipysrc\pandas\web\pandas_web.py`

```
"""
Simple static site generator for the pandas web.

pandas_web.py takes a directory as parameter, and copies all the files into the
target directory after converting markdown files into html and rendering both
markdown and html files with a context. The context is obtained by parsing
the file ``config.yml`` in the root of the source directory.

The file should contain:

main:
  template_path: <path_to_the_jinja2_templates_directory>
  base_template: <template_file_all_other_files_will_extend>
  ignore:
  - <list_of_files_in_the_source_that_will_not_be_copied>
  github_repo_url: <organization/repo-name>
  context_preprocessors:
  - <list_of_functions_that_will_enrich_the_context_parsed_in_this_file>
  markdown_extensions:
  - <list_of_markdown_extensions_that_will_be_loaded>


The rest of the items in the file will be added directly to the context.
"""

import argparse
import collections
import datetime
import importlib
import itertools
import json
import operator
import os
import pathlib
import re
import shutil
import sys
import time
import typing

import feedparser
import jinja2
import markdown
from packaging import version
import requests
import yaml

# 从环境变量中获取 GitHub API Token
api_token = os.environ.get("GITHUB_TOKEN")

# 如果获取到了 API Token，设置请求头部，包含 Token 以进行授权
if api_token is not None:
    GITHUB_API_HEADERS = {"Authorization": f"Bearer {api_token}"}
else:
    # 如果未获取到 API Token，设置请求头部为空
    GITHUB_API_HEADERS = {}

class Preprocessors:
    """
    Built-in context preprocessors.

    Context preprocessors are functions that receive the context used to
    render the templates, and enriches it with additional information.

    The original context is obtained by parsing ``config.yml``, and
    anything else needed just be added with context preprocessors.
    """

    @staticmethod
    def current_year(context):
        """
        Add the current year to the context, so it can be used for the copyright
        note, or other places where it is needed.
        """
        context["current_year"] = datetime.datetime.now().year
        return context

    @staticmethod
    def navbar_add_info(context):
        """
        Items in the main navigation bar can be direct links, or dropdowns with
        subitems. This context preprocessor adds a boolean field
        ``has_subitems`` that tells which one of them every element is. It
        also adds a ``slug`` field to be used as a CSS id.
        """
        # 遍历导航栏项目，为每个项目添加额外的信息字段
        for i, item in enumerate(context["navbar"]):
            context["navbar"][i] = dict(
                item,
                has_subitems=isinstance(item["target"], list),  # 判断是否有子项目
                slug=(item["name"].replace(" ", "-").lower()),  # 创建用于 CSS id 的 slug
            )
        return context
    def maintainers_add_info(context):
        """
        给定 yaml 文件中定义的活跃维护者，获取他们的 GitHub 用户信息。
        """
        # 找出同时在活跃维护者和非活跃维护者列表中的重复项
        repeated = set(context["maintainers"]["active"]) & set(
            context["maintainers"]["inactive"]
        )
        if repeated:
            raise ValueError(f"Maintainers {repeated} are both active and inactive")

        # 初始化维护者信息的空字典
        maintainers_info = {}
        # 遍历所有活跃和非活跃维护者
        for user in (
            context["maintainers"]["active"] + context["maintainers"]["inactive"]
        ):
            # 向 GitHub API 发送 GET 请求，获取用户信息
            resp = requests.get(
                f"https://api.github.com/users/{user}",
                headers=GITHUB_API_HEADERS,
                timeout=5,
            )
            # 如果 API 返回状态码为 403，超出配额限制
            if resp.status_code == 403:
                sys.stderr.write(
                    "WARN: GitHub API quota exceeded when fetching maintainers\n"
                )
                # 使用网站上保存的维护者信息备用
                resp_bkp = requests.get(
                    context["main"]["production_url"] + "maintainers.json", timeout=5
                )
                resp_bkp.raise_for_status()
                maintainers_info = resp_bkp.json()
                break

            # 如果请求成功，将用户信息保存到字典中
            resp.raise_for_status()
            maintainers_info[user] = resp.json()

        # 将 GitHub 用户信息存储到上下文中
        context["maintainers"]["github_info"] = maintainers_info

        # 将从 GitHub 获取的数据保存，以备将来超出 GitHub API 配额时使用
        with open(
            pathlib.Path(context["target_path"]) / "maintainers.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(maintainers_info, f)

        # 返回更新后的上下文对象
        return context

    @staticmethod
    # 定义一个静态方法用于向上下文中添加发布信息
    def home_add_releases(context):
        # 初始化一个空列表用于存放发布信息
        context["releases"] = []

        # 从上下文中获取 GitHub 仓库的 URL
        github_repo_url = context["main"]["github_repo_url"]
        # 发起 GET 请求获取 GitHub 仓库的发布信息
        resp = requests.get(
            f"https://api.github.com/repos/{github_repo_url}/releases",
            headers=GITHUB_API_HEADERS,
            timeout=5,
        )
        # 处理 GitHub API 响应状态码为 403 的情况
        if resp.status_code == 403:
            sys.stderr.write("WARN: GitHub API quota exceeded when fetching releases\n")
            # 发起备用请求获取发布信息
            resp_bkp = requests.get(
                context["main"]["production_url"] + "releases.json", timeout=5
            )
            resp_bkp.raise_for_status()
            # 解析备用请求的 JSON 数据作为发布信息
            releases = resp_bkp.json()
        else:
            # 处理正常情况下的 GitHub API 响应
            resp.raise_for_status()
            # 解析 GitHub API 返回的 JSON 数据作为发布信息
            releases = resp.json()

        # 将获取到的发布信息写入到目标路径下的 releases.json 文件中
        with open(
            pathlib.Path(context["target_path"]) / "releases.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(releases, f, default=datetime.datetime.isoformat)

        # 遍历处理每一个发布信息对象
        for release in releases:
            # 跳过预发布版本的处理
            if release["prerelease"]:
                continue
            # 解析并转换发布时间格式为 datetime 对象
            published = datetime.datetime.strptime(
                release["published_at"], "%Y-%m-%dT%H:%M:%SZ"
            )
            # 向上下文的 releases 列表中添加格式化后的发布信息
            context["releases"].append(
                {
                    "name": release["tag_name"].lstrip("v"),  # 去除版本号前的 'v' 字符
                    "parsed_version": version.parse(release["tag_name"].lstrip("v")),  # 解析版本号为版本对象
                    "tag": release["tag_name"],  # 添加原始标签名
                    "published": published,  # 添加发布时间
                    "url": (
                        release["assets"][0]["browser_download_url"]  # 获取第一个资源的下载链接
                        if release["assets"]
                        else ""  # 若无资源则为空字符串
                    ),
                }
            )

        # 使用 itertools.groupby 根据主版本号和次版本号分组发布信息
        grouped_releases = itertools.groupby(
            context["releases"],
            key=lambda r: (r["parsed_version"].major, r["parsed_version"].minor),
        )
        # 对每个分组选取次版本号最大的发布版本
        context["releases"] = [
            max(release_group, key=lambda r: r["parsed_version"].minor)
            for _, release_group in grouped_releases
        ]
        # 按照版本号降序排序发布信息列表
        context["releases"].sort(key=lambda r: r["parsed_version"], reverse=True)

        # 返回更新后的上下文对象
        return context
# 定义函数，根据字符串表示的对象路径获取对应的 Python 对象
def get_callable(obj_as_str: str) -> object:
    """
    Get a Python object from its string representation.

    For example, for ``sys.stdout.write`` would import the module ``sys``
    and return the ``write`` function.
    """
    # 按照点号分割字符串表示的对象路径
    components = obj_as_str.split(".")
    attrs = []
    # 逐层尝试导入模块，直到找到有效的模块对象或耗尽路径
    while components:
        try:
            obj = importlib.import_module(".".join(components))
        except ImportError:
            # 将未能导入的部分添加到属性列表中
            attrs.insert(0, components.pop())
        else:
            break

    # 如果未成功导入任何模块，抛出 ImportError 异常
    if not obj:
        raise ImportError(f'Could not import "{obj_as_str}"')

    # 依次获取属性对象
    for attr in attrs:
        obj = getattr(obj, attr)

    return obj


# 定义函数，加载配置文件并生成上下文信息，其中包括预处理器的处理结果
def get_context(config_fname: str, **kwargs):
    """
    Load the config yaml as the base context, and enrich it with the
    information added by the context preprocessors defined in the file.
    """
    # 使用 UTF-8 编码打开配置文件
    with open(config_fname, encoding="utf-8") as f:
        # 解析 YAML 格式的配置文件内容
        context = yaml.safe_load(f)

    # 将配置文件所在路径添加到上下文中
    context["source_path"] = os.path.dirname(config_fname)
    # 更新上下文信息，包括传入的关键字参数
    context.update(kwargs)

    # 获取主配置中定义的上下文预处理器，并依次执行
    preprocessors = (
        get_callable(context_prep)
        for context_prep in context["main"]["context_preprocessors"]
    )
    for preprocessor in preprocessors:
        # 调用每个预处理器函数处理上下文信息
        context = preprocessor(context)
        # 检查预处理器函数是否正确返回了结果
        msg = f"{preprocessor.__name__} is missing the return statement"
        assert context is not None, msg

    return context


# 定义生成器函数，生成源目录中的所有文件路径
def get_source_files(source_path: str) -> typing.Generator[str, None, None]:
    """
    Generate the list of files present in the source directory.
    """
    # 递归遍历源目录中的所有文件
    for root, dirs, fnames in os.walk(source_path):
        # 计算每个文件相对于源目录的路径
        root_rel_path = os.path.relpath(root, source_path)
        for fname in fnames:
            # 生成每个文件的完整路径并返回
            yield os.path.join(root_rel_path, fname)


# 定义函数，将内容包装成扩展基础模板的形式，以便用 Jinja2 渲染
def extend_base_template(content: str, base_template: str) -> str:
    """
    Wrap document to extend the base template, before it is rendered with
    Jinja2.
    """
    # 构建包含继承和内容块定义的模板字符串
    result = '{% extends "' + base_template + '" %}'
    result += "{% block body %}"
    result += content
    result += "{% endblock %}"
    return result


# 定义主函数，实现将源目录中的文件复制到目标目录的功能
def main(
    source_path: str,
    target_path: str,
) -> int:
    """
    Copy every file in the source directory to the target directory.

    For ``.md`` and ``.html`` files, render them with the context
    before copying them. ``.md`` files are transformed to HTML.
    """
    # 构建配置文件路径
    config_fname = os.path.join(source_path, "config.yml")

    # 清空目标目录并创建新目录结构
    shutil.rmtree(target_path, ignore_errors=True)
    os.makedirs(target_path, exist_ok=True)

    # 输出提示信息
    sys.stderr.write("Generating context...\n")
    # 加载配置文件并生成上下文信息
    context = get_context(config_fname, target_path=target_path)
    # 输出提示信息
    sys.stderr.write("Context generated\n")

    # 获取模板文件路径
    templates_path = os.path.join(source_path, context["main"]["templates_path"])
    # 创建 Jinja2 环境对象
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(templates_path))
    # 遍历源路径下的所有文件名
    for fname in get_source_files(source_path):
        # 检查文件名是否在上下文中的忽略列表中，如果是则跳过当前循环
        if os.path.normpath(fname) in context["main"]["ignore"]:
            continue
    
        # 在标准错误流中输出正在处理的文件名信息
        sys.stderr.write(f"Processing {fname}\n")
    
        # 获取当前文件名的目录名
        dirname = os.path.dirname(fname)
    
        # 确保目标路径下的目录结构与当前文件的目录结构相匹配，如果目录不存在则创建
        os.makedirs(os.path.join(target_path, dirname), exist_ok=True)
    
        # 获取文件扩展名
        extension = os.path.splitext(fname)[-1]
    
        # 如果文件扩展名是 .html 或 .md
        if extension in (".html", ".md"):
            # 打开文件并读取内容
            with open(os.path.join(source_path, fname), encoding="utf-8") as f:
                content = f.read()
    
            # 如果文件扩展名是 .md，则将内容转换为 Markdown 格式
            if extension == ".md":
                body = markdown.markdown(
                    content, extensions=context["main"]["markdown_extensions"]
                )
    
                # 手动应用 Bootstrap 的表格格式
                # Python-Markdown 不允许手动配置表格属性
                body = body.replace("<table>", '<table class="table table-bordered">')
    
                # 将 Markdown 转换后的内容扩展为基础模板
                content = extend_base_template(body, context["main"]["base_template"])
    
            # 设置上下文中的 base_url 属性，基于文件名的目录结构层次
            context["base_url"] = "".join(["../"] * os.path.normpath(fname).count("/"))
    
            # 使用 Jinja2 模板引擎渲染内容
            content = jinja_env.from_string(content).render(**context)
    
            # 将文件名的扩展名改为 .html
            fname_html = os.path.splitext(fname)[0] + ".html"
    
            # 将渲染后的内容写入目标路径中的 .html 文件
            with open(
                os.path.join(target_path, fname_html), "w", encoding="utf-8"
            ) as f:
                f.write(content)
    
        # 如果文件扩展名不是 .html 或 .md，则直接复制文件
        else:
            shutil.copy(
                os.path.join(source_path, fname), os.path.join(target_path, dirname)
            )
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则执行以下代码块

    # 创建一个参数解析器对象，用于处理命令行参数
    parser = argparse.ArgumentParser(description="Documentation builder.")

    # 添加一个位置参数，指定源目录的路径（必须包含 config.yml 文件）
    parser.add_argument(
        "source_path", help="path to the source directory (must contain config.yml)"
    )

    # 添加一个可选参数，指定输出目录的路径，默认为 "build"
    parser.add_argument(
        "--target-path", default="build", help="directory where to write the output"
    )

    # 解析命令行参数，并将其存储到 args 变量中
    args = parser.parse_args()

    # 调用 main 函数，并传入解析后的参数中的源路径和目标路径
    sys.exit(main(args.source_path, args.target_path))
```