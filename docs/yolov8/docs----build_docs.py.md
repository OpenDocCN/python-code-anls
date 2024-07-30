# `.\yolov8\docs\build_docs.py`

```py
# 设置环境变量以解决 DeprecationWarning：Jupyter 正在迁移到使用标准的 platformdirs
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

# 定义常量 DOCS 为当前脚本所在目录的父目录的绝对路径
DOCS = Path(__file__).parent.resolve()

# 定义常量 SITE 为 DOCS 的父目录下的 'site' 目录的路径
SITE = DOCS.parent / "site"

def prepare_docs_markdown(clone_repos=True):
    """使用 mkdocs 构建文档。"""
    # 如果 SITE 目录存在，则删除已有的 SITE 目录
    if SITE.exists():
        print(f"Removing existing {SITE}")
        shutil.rmtree(SITE)

    # 如果 clone_repos 为 True，则获取 hub-sdk 仓库
    if clone_repos:
        repo = "https://github.com/ultralytics/hub-sdk"
        local_dir = DOCS.parent / Path(repo).name
        # 如果本地目录不存在，则克隆仓库
        if not local_dir.exists():
            os.system(f"git clone {repo} {local_dir}")
        # 更新仓库
        os.system(f"git -C {local_dir} pull")
        # 如果存在，则删除现有的 'en/hub/sdk' 目录
        shutil.rmtree(DOCS / "en/hub/sdk", ignore_errors=True)
        # 拷贝仓库中的 'docs' 目录到 'en/hub/sdk'
        shutil.copytree(local_dir / "docs", DOCS / "en/hub/sdk")
        # 如果存在，则删除现有的 'hub_sdk' 目录
        shutil.rmtree(DOCS.parent / "hub_sdk", ignore_errors=True)
        # 拷贝仓库中的 'hub_sdk' 目录到 'hub_sdk'
        shutil.copytree(local_dir / "hub_sdk", DOCS.parent / "hub_sdk")
        print(f"Cloned/Updated {repo} in {local_dir}")

    # 对 'en' 目录下的所有 '.md' 文件添加 frontmatter
    for file in tqdm((DOCS / "en").rglob("*.md"), desc="Adding frontmatter"):
        update_markdown_files(file)

def update_page_title(file_path: Path, new_title: str):
    """更新 HTML 文件的标题。"""
    # 打开文件并读取其内容，使用 UTF-8 编码
    with open(file_path, encoding="utf-8") as file:
        content = file.read()

    # 使用正则表达式替换现有标题为新标题
    updated_content = re.sub(r"<title>.*?</title>", f"<title>{new_title}</title>", content)

    # 将更新后的内容写回文件
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_content)
def update_html_head(script=""):
    """Update the HTML head section of each file."""
    # 获取指定目录下所有的 HTML 文件
    html_files = Path(SITE).rglob("*.html")
    # 遍历每个 HTML 文件
    for html_file in tqdm(html_files, desc="Processing HTML files"):
        # 打开文件并读取 HTML 内容
        with html_file.open("r", encoding="utf-8") as file:
            html_content = file.read()

        # 如果指定的脚本已经在 HTML 文件中，则直接返回
        if script in html_content:  # script already in HTML file
            return

        # 查找 HTML 头部结束标签的索引位置
        head_end_index = html_content.lower().rfind("</head>")
        if head_end_index != -1:
            # 在头部结束标签之前插入指定的 JavaScript 脚本
            new_html_content = html_content[:head_end_index] + script + html_content[head_end_index:]
            # 将更新后的 HTML 内容写回文件
            with html_file.open("w", encoding="utf-8") as file:
                file.write(new_html_content)


def update_subdir_edit_links(subdir="", docs_url=""):
    """Update the HTML head section of each file."""
    # 如果子目录以斜杠开头，则去掉斜杠
    if str(subdir[0]) == "/":
        subdir = str(subdir[0])[1:]
    # 获取子目录下所有的 HTML 文件
    html_files = (SITE / subdir).rglob("*.html")
    # 遍历每个 HTML 文件
    for html_file in tqdm(html_files, desc="Processing subdir files"):
        # 打开文件并解析为 BeautifulSoup 对象
        with html_file.open("r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")

        # 查找包含特定类和标题的锚点标签，并更新其 href 属性
        a_tag = soup.find("a", {"class": "md-content__button md-icon"})
        if a_tag and a_tag["title"] == "Edit this page":
            a_tag["href"] = f"{docs_url}{a_tag['href'].split(subdir)[-1]}"

        # 将更新后的 HTML 内容写回文件
        with open(html_file, "w", encoding="utf-8") as file:
            file.write(str(soup))


def update_markdown_files(md_filepath: Path):
    """Creates or updates a Markdown file, ensuring frontmatter is present."""
    # 如果 Markdown 文件存在，则读取其内容
    if md_filepath.exists():
        content = md_filepath.read_text().strip()

        # 替换特定的书名号字符为直引号
        content = content.replace("‘", "'").replace("’", "'")

        # 如果前置内容缺失，则添加默认的前置内容
        if not content.strip().startswith("---\n"):
            header = "---\ncomments: true\ndescription: TODO ADD DESCRIPTION\nkeywords: TODO ADD KEYWORDS\n---\n\n"
            content = header + content

        # 确保 MkDocs admonitions "=== " 开始的行前后有空行
        lines = content.split("\n")
        new_lines = []
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith("=== "):
                if i > 0 and new_lines[-1] != "":
                    new_lines.append("")
                new_lines.append(line)
                if i < len(lines) - 1 and lines[i + 1].strip() != "":
                    new_lines.append("")
            else:
                new_lines.append(line)
        content = "\n".join(new_lines)

        # 如果文件末尾缺少换行符，则添加一个
        if not content.endswith("\n"):
            content += "\n"

        # 将更新后的内容写回 Markdown 文件
        md_filepath.write_text(content)
    return


def update_docs_html():
    # 此函数可能尚未实现具体功能，暂无需添加注释
    pass
    """Updates titles, edit links, head sections, and converts plaintext links in HTML documentation."""
    # 更新页面标题
    update_page_title(SITE / "404.html", new_title="Ultralytics Docs - Not Found")

    # 更新编辑链接
    update_subdir_edit_links(
        subdir="hub/sdk/",  # 不要使用开头的斜杠
        docs_url="https://github.com/ultralytics/hub-sdk/tree/main/docs/",
    )

    # 将纯文本链接转换为 HTML 超链接
    files_modified = 0
    for html_file in tqdm(SITE.rglob("*.html"), desc="Converting plaintext links"):
        # 打开 HTML 文件并读取内容
        with open(html_file, "r", encoding="utf-8") as file:
            content = file.read()
        # 将纯文本链接转换为 HTML 格式
        updated_content = convert_plaintext_links_to_html(content)
        # 如果内容有更新，则写入更新后的内容
        if updated_content != content:
            with open(html_file, "w", encoding="utf-8") as file:
                file.write(updated_content)
            # 记录已修改的文件数
            files_modified += 1
    # 打印修改的文件数
    print(f"Modified plaintext links in {files_modified} files.")

    # 更新 HTML 文件的 head 部分
    script = ""
    # 如果有脚本内容，则更新 HTML 的 head 部分
    if any(script):
        update_html_head(script)
def convert_plaintext_links_to_html(content):
    """Converts plaintext links to HTML hyperlinks in the main content area only."""
    # 使用BeautifulSoup解析传入的HTML内容
    soup = BeautifulSoup(content, "html.parser")

    # 查找主要内容区域（根据HTML结构调整选择器）
    main_content = soup.find("main") or soup.find("div", class_="md-content")
    if not main_content:
        return content  # 如果找不到主内容区域，则返回原始内容

    modified = False
    # 遍历主内容区域中的段落和列表项
    for paragraph in main_content.find_all(["p", "li"]):
        for text_node in paragraph.find_all(string=True, recursive=False):
            # 忽略链接和代码块的父节点
            if text_node.parent.name not in {"a", "code"}:
                # 使用正则表达式将文本节点中的链接转换为HTML超链接
                new_text = re.sub(
                    r'(https?://[^\s()<>]+(?:\.[^\s()<>]+)+)(?<![.,:;\'"])',
                    r'<a href="\1">\1</a>',
                    str(text_node),
                )
                # 如果生成了新的<a>标签，则替换原文本节点
                if "<a" in new_text:
                    new_soup = BeautifulSoup(new_text, "html.parser")
                    text_node.replace_with(new_soup)
                    modified = True

    # 如果修改了内容，则返回修改后的HTML字符串；否则返回原始内容
    return str(soup) if modified else content


def main():
    """Builds docs, updates titles and edit links, and prints local server command."""
    prepare_docs_markdown()

    # 构建主文档
    print(f"Building docs from {DOCS}")
    subprocess.run(f"mkdocs build -f {DOCS.parent}/mkdocs.yml --strict", check=True, shell=True)
    print(f"Site built at {SITE}")

    # 更新文档的HTML页面
    update_docs_html()

    # 显示用于启动本地服务器的命令
    print('Docs built correctly ✅\nServe site at http://localhost:8000 with "python -m http.server --directory site"')


if __name__ == "__main__":
    main()
```