# `D:\src\scipysrc\sympy\release\github_release.py`

```
#!/usr/bin/env python

import os  # 导入操作系统相关的模块
import json  # 导入处理 JSON 格式数据的模块
from subprocess import check_output  # 导入执行外部命令并返回输出结果的函数
from collections import OrderedDict, defaultdict  # 导入有序字典和默认字典
from collections.abc import Mapping  # 导入抽象基类中的 Mapping 类
import glob  # 导入文件名匹配模块
from contextlib import contextmanager  # 导入上下文管理相关的模块

import requests  # 导入 HTTP 请求库
from requests_oauthlib import OAuth2  # 导入 OAuth2 认证相关库


def main(version, push=None):
    """
    WARNING: If push is given as --push then this will push the release to
    github.
    """
    push = push == '--push'  # 根据传入参数判断是否需要推送到 GitHub
    _GitHub_release(version, push)


def error(msg):
    raise ValueError(msg)  # 抛出 ValueError 异常并输出指定错误消息


def blue(text):
    return "\033[34m%s\033[0m" % text  # 返回带有蓝色 ANSI 色彩的文本


def red(text):
    return "\033[31m%s\033[0m" % text  # 返回带有红色 ANSI 色彩的文本


def green(text):
    return "\033[32m%s\033[0m" % text  # 返回带有绿色 ANSI 色彩的文本


def _GitHub_release(version, push, username=None, user='sympy', token=None,
    token_file_path="~/.sympy/release-token", repo='sympy', draft=False):
    """
    Upload the release files to GitHub.

    The tag must be pushed up first. You can test on another repo by changing
    user and repo.
    """
    if not requests:
        error("requests and requests-oauthlib must be installed to upload to GitHub")  # 如果 requests 和 requests-oauthlib 未安装，则输出错误信息

    release_text = GitHub_release_text(version)  # 获取发布的文本内容
    short_version = get_sympy_short_version(version)  # 获取 SymPy 的简短版本号
    tag = 'sympy-' + version  # 设置 GitHub 标签
    prerelease = short_version != version  # 检查是否为预发布版本

    urls = URLs(user=user, repo=repo)  # 创建 URLs 实例，指定 GitHub 用户名和仓库名
    if not username:
        username = input("GitHub username: ")  # 如果未提供用户名，则从用户输入获取
    token = load_token_file(token_file_path)  # 从指定路径加载 GitHub 认证令牌
    if not token:
        username, password, token = GitHub_authenticate(urls, username, token)  # 进行 GitHub 认证流程获取令牌

    # 如果标签尚未推送，则终止发布过程
    if not check_tag_exists(version):
        sys.exit(red("The tag for this version has not been pushed yet. Cannot upload the release."))

    # 参考 https://developer.github.com/v3/repos/releases/#create-a-release
    # 首先创建发布
    post = {}
    post['tag_name'] = tag  # 设置发布的标签名称
    post['name'] = "SymPy " + version  # 设置发布的名称
    post['body'] = release_text  # 设置发布的正文内容
    post['draft'] = draft  # 设置是否为草稿
    post['prerelease'] = prerelease  # 设置是否为预发布版本

    print("Creating release for tag", tag, end=' ')

    if push:
        result = query_GitHub(urls.releases_url, username, password=None,
            token=token, data=json.dumps(post)).json()  # 如果需要推送，则向 GitHub 发送发布请求并获取结果
        release_id = result['id']  # 获取发布的唯一标识符
    else:
        print(green("Not pushing!"))  # 如果不需要推送，则提示未推送

    print(green("Done"))  # 输出发布完成的提示信息

    # 然后，上传所有文件至发布页面。
    # 遍历描述列表中的每个项目
    for key in descriptions:
        # 获取当前项目的压缩文件名，根据版本号
        tarball = get_tarball_name(key, version)

        # 初始化参数字典
        params = {}
        # 设置参数字典中的'name'键为当前压缩文件名
        params['name'] = tarball

        # 根据压缩文件名的后缀设置不同的HTTP头部内容类型
        if tarball.endswith('gz'):
            headers = {'Content-Type':'application/gzip'}
        elif tarball.endswith('pdf'):
            headers = {'Content-Type':'application/pdf'}
        elif tarball.endswith('zip'):
            headers = {'Content-Type':'application/zip'}
        else:
            headers = {'Content-Type':'application/octet-stream'}

        # 输出当前正在上传的文件名
        print("Uploading", tarball, end=' ')
        # 刷新标准输出，确保立即显示
        sys.stdout.flush()

        # 使用文件操作符打开当前文件以二进制只读模式
        with open(os.path.join('release/release-' + version, tarball), 'rb') as f:
            # 如果push标志为真，执行GitHub查询和上传操作
            if push:
                # 调用query_GitHub函数向GitHub的上传URL发送请求，使用给定的用户名、token和数据
                result = query_GitHub(urls.release_uploads_url % release_id, username,
                    password=None, token=token, data=f, params=params,
                    headers=headers).json()
            else:
                # 如果push标志为假，输出不上传的提示信息
                print(green("Not uploading!"))

        # 输出上传完成的提示信息
        print(green("Done"))

    # TODO: download the files and check that they have the right sha256 sum
# 生成适用于 GitHub 发布 Markdown 框的文本
def GitHub_release_text(version):
    # 获取 SymPy 的简短版本号
    shortversion = get_sympy_short_version(version)
    # 生成包含版本信息的 HTML 表格
    htmltable = table(version)
    # 构建发布注释文本，包含版本链接和 HTML 表格
    out = """\
See https://github.com/sympy/sympy/wiki/release-notes-for-{shortversion} for the release notes.

{htmltable}

**Note**: Do not download the **Source code (zip)** or the **Source code (tar.gz)**
files below.
"""
    # 格式化输出字符串，替换版本和表格内容
    out = out.format(shortversion=shortversion, htmltable=htmltable)
    # 打印蓝色提示，复制发布注释到 GitHub 发布表单中
    print(blue("Here are the release notes to copy into the GitHub release "
        "Markdown form:"))
    print()
    print(out)
    # 返回构建的发布注释文本
    return out


# 获取发布的 SymPy 短版本号，不包括任何 rc 标签（如 0.7.3）
def get_sympy_short_version(version):
    # 将版本号分割成部分
    parts = version.split('.')
    # 移除尾部的 rc 标签，例如 1.10rc1 -> [1, 10]
    lastpart = ''
    for dig in parts[-1]:
        if dig.isdigit():
            lastpart += dig
        else:
            break
    parts[-1] = lastpart
    # 返回修正后的版本号字符串
    return '.'.join(parts)


class URLs(object):
    """
    This class contains URLs and templates which used in requests to GitHub API
    """

    # 初始化 URLs 类，生成所有的 URL 和模板
    def __init__(self, user="sympy", repo="sympy",
        api_url="https://api.github.com",
        authorize_url="https://api.github.com/authorizations",
        uploads_url='https://uploads.github.com',
        main_url='https://github.com'):
        """
        Generates all URLs and templates
        """
        # 设置 GitHub 用户名、仓库名以及 API 和授权 URL
        self.user = user
        self.repo = repo
        self.api_url = api_url
        self.authorize_url = authorize_url
        self.uploads_url = uploads_url
        self.main_url = main_url

        # 构建特定仓库的 Pull 请求列表 URL
        self.pull_list_url = api_url + "/repos" + "/" + user + "/" + repo + "/pulls"
        # 构建特定仓库的问题列表 URL
        self.issue_list_url = api_url + "/repos/" + user + "/" + repo + "/issues"
        # 构建特定仓库的发布列表 URL
        self.releases_url = api_url + "/repos/" + user + "/" + repo + "/releases"
        # 构建单个问题的模板 URL
        self.single_issue_template = self.issue_list_url + "/%d"
        # 构建单个 Pull 请求的模板 URL
        self.single_pull_template = self.pull_list_url + "/%d"
        # 构建用户信息的模板 URL
        self.user_info_template = api_url + "/users/%s"
        # 构建用户仓库列表的模板 URL
        self.user_repos_template = api_url + "/users/%s/repos"
        # 构建问题评论的模板 URL
        self.issue_comment_template = (api_url + "/repos" + "/" + user + "/" + repo + "/issues/%d" +
            "/comments")
        # 构建发布上传文件的 URL 模板
        self.release_uploads_url = (uploads_url + "/repos/" + user + "/" +
            repo + "/releases/%d" + "/assets")
        # 构建发布下载的 URL 模板
        self.release_download_url = (main_url + "/" + user + "/" + repo +
            "/releases/download/%s/%s")


# 加载 Token 文件，用于 GitHub 发布
def load_token_file(path="~/.sympy/release-token"):
    # 打印正在使用的 Token 文件路径
    print("> Using token file %s" % path)

    # 展开和规范化 Token 文件路径
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    # 如果 Token 文件存在，则尝试读取第一行的 Token
    if os.path.isfile(path):
        try:
            with open(path) as f:
                token = f.readline()
        except IOError:
            # 如果无法读取 Token 文件，则输出错误信息
            print("> Unable to read token file")
            return
    else:
        # 如果 Token 文件不存在，则输出提示信息
        print("> Token file does not exist")
        return

    # 返回读取到的 Token，去除首尾空格
    return token.strip()
# GitHub 身份验证函数，用于验证用户的 GitHub 账号信息
def GitHub_authenticate(urls, username, token=None):
    # 登录信息提示消息
    _login_message = """\
Enter your GitHub username & password or press ^C to quit. The password
will be kept as a Python variable as long as this script is running and
https to authenticate with GitHub, otherwise not saved anywhere else:\
"""
    # 如果已提供用户名，则显示认证中的用户名
    if username:
        print("> Authenticating as %s" % username)
    else:
        # 否则打印登录信息提示消息，并等待用户输入用户名
        print(_login_message)
        username = input("Username: ")

    authenticated = False

    # 如果提供了 token，则使用 token 进行认证
    if token:
        print("> Authenticating using token")
        try:
            # 调用 GitHub_check_authentication 函数验证 token
            GitHub_check_authentication(urls, username, None, token)
        except AuthenticationFailed:
            print(">     Authentication failed")
        else:
            print(">     OK")
            password = None
            authenticated = True

    # 如果未认证成功，则循环直到认证成功
    while not authenticated:
        # 获取用户输入的密码
        password = getpass("Password: ")
        try:
            print("> Checking username and password ...")
            # 调用 GitHub_check_authentication 函数验证用户名和密码
            GitHub_check_authentication(urls, username, password, None)
        except AuthenticationFailed:
            print(">     Authentication failed")
        else:
            print(">     OK.")
            authenticated = True

    # 如果成功获取密码，则询问是否生成 API token
    if password:
        generate = input("> Generate API token? [Y/n] ")
        if generate.lower() in ["y", "ye", "yes", ""]:
            # 如果用户同意生成 token，则询问 token 的名称，默认为 "SymPy Release"
            name = input("> Name of token on GitHub? [SymPy Release] ")
            if name == "":
                name = "SymPy Release"
            # 调用 generate_token 函数生成 token
            token = generate_token(urls, username, password, name=name)
            print("Your token is", token)
            print("Use this token from now on as GitHub_release:token=" + token +
                ",username=" + username)
            print(red("DO NOT share this token with anyone"))
            # 询问用户是否将 token 保存到文件中
            save = input("Do you want to save this token to a file [yes]? ")
            if save.lower().strip() in ['y', 'yes', 'ye', '']:
                # 如果用户同意，调用 save_token_file 函数保存 token 到文件中
                save_token_file(token)

    # 返回认证成功的用户名、密码和 token（如果有）
    return username, password, token


def run(*cmdline, cwd=None):
    """
    Run command in subprocess and get lines of output
    """
    # 在子进程中运行命令并获取输出的行列表
    return check_output(cmdline, encoding='utf-8', cwd=cwd).splitlines()


def check_tag_exists(version):
    """
    Check if the tag for this release has been uploaded yet.
    """
    # 检查是否已经上传了此版本的标签
    tag = 'sympy-' + version
    all_tag_lines = run('git', 'ls-remote', '--tags', 'origin')
    return any(tag in tag_line for tag_line in all_tag_lines)


def generate_token(urls, username, password, OTP=None, name="SymPy Release"):
    # 准备需要加密的 JSON 数据，包括 token 的作用范围和注释
    enc_data = json.dumps(
        {
            "scopes": ["public_repo"],
            "note": name
        }
    )

    # 使用 GitHub API 进行用户认证，并获取 token
    url = urls.authorize_url
    rep = query_GitHub(url, username=username, password=password,
        data=enc_data).json()
    return rep["token"]


def GitHub_check_authentication(urls, username, password, token):
    """
    Checks that username & password is valid.
    """
    # 调用 query_GitHub 函数，验证 GitHub 账号信息的有效性
    query_GitHub(urls.api_url, username, password, token)


class AuthenticationFailed(Exception):
    # 认证失败的异常类
    pass
# 定义一个函数用于查询 GitHub API
def query_GitHub(url, username=None, password=None, token=None, data=None,
    OTP=None, headers=None, params=None, files=None):
    """
    Query GitHub API.

    In case of a multipage result, DOES NOT query the next page.

    """
    # 如果 headers 为 None，则初始化为空字典
    headers = headers or {}

    # 如果存在 OTP，将其添加到 headers 中的 X-GitHub-OTP 字段
    if OTP:
        headers['X-GitHub-OTP'] = OTP

    # 根据 token 的有无，选择不同的认证方式
    if token:
        auth = OAuth2(client_id=username, token={"access_token": token,
            "token_type": 'bearer'})
    else:
        auth = HTTPBasicAuth(username, password)

    # 根据是否有 data 参数，选择使用 POST 或 GET 请求
    if data:
        r = requests.post(url, auth=auth, data=data, headers=headers,
            params=params, files=files)
    else:
        r = requests.get(url, auth=auth, headers=headers, params=params, stream=True)

    # 如果返回的状态码为 401，处理认证失败的情况
    if r.status_code == 401:
        # 获取是否需要两步验证，若需要则提示用户输入验证码
        two_factor = r.headers.get('X-GitHub-OTP')
        if two_factor:
            print("A two-factor authentication code is required:", two_factor.split(';')[1].strip())
            OTP = input("Authentication code: ")
            # 重新调用自身，用输入的 OTP 再次进行 GitHub API 查询
            return query_GitHub(url, username=username, password=password,
                token=token, data=data, OTP=OTP)

        # 若未返回需要两步验证的信息，则抛出认证失败的异常
        raise AuthenticationFailed("invalid username or password")

    # 若请求成功，抛出任何请求错误
    r.raise_for_status()
    return r


# 定义一个函数用于保存 token 到文件中
def save_token_file(token):
    # 提示用户输入 token 文件的保存位置，默认为 "~/.sympy/release-token"
    token_file = input("> Enter token file location [~/.sympy/release-token] ")
    token_file = token_file or "~/.sympy/release-token"

    # 将输入的 token 文件路径扩展并转换为绝对路径
    token_file_expand = os.path.expanduser(token_file)
    token_file_expand = os.path.abspath(token_file_expand)
    token_folder, _ = os.path.split(token_file_expand)

    try:
        # 如果 token 文件夹不存在，则创建它，设置权限为 700
        if not os.path.isdir(token_folder):
            os.mkdir(token_folder, 0o700)
        
        # 将 token 写入文件中
        with open(token_file_expand, 'w') as f:
            f.write(token + '\n')

        # 设置 token 文件的权限为只读和只写
        os.chmod(token_file_expand, stat.S_IREAD | stat.S_IWRITE)
    
    # 处理可能的异常情况
    except OSError as e:
        print("> Unable to create folder for token file: ", e)
        return
    except IOError as e:
        print("> Unable to save token file: ", e)
        return

    # 返回成功保存的 token 文件路径
    return token_file


# 定义一个函数用于生成版本信息的 HTML 表格
def table(version):
    """
    Make an html table of the downloads.

    This is for pasting into the GitHub releases page. See GitHub_release().
    """
    # 获取版本相关的 tarball 格式字典和短版本号
    tarball_formatter_dict = dict(_tarball_format(version))
    shortversion = get_sympy_short_version(version)

    # 将版本号添加到 tarball_formatter_dict 字典中
    tarball_formatter_dict['version'] = shortversion

    # 获取版本相关的 SHA256 校验和和文件名的字典
    sha256s = [i.split('\t') for i in _sha256(version, print_=False, local=True).split('\n')]
    sha256s_dict = {name: sha256 for sha256, name in sha256s}

    # 获取版本相关的文件大小和文件名的字典
    sizes = [i.split('\t') for i in _size(version, print_=False).split('\n')]
    sizes_dict = {name: size for size, name in sizes}

    # 初始化一个空的表格
    table = []

    # 定义一个上下文管理器函数用于生成 HTML 标签，添加到 table 列表中
    @contextmanager
    def tag(name):
        table.append("<%s>" % name)
        yield
        table.append("</%s>" % name)
    @contextmanager
    # 定义一个生成器函数 a_href，用于生成包含指定链接的 HTML 锚标签
    def a_href(link):
        # 将包含链接的 HTML 锚标签添加到 table 列表中
        table.append("<a href=\"%s\">" % link)
        # 生成器函数的 yield 语句，用于生成标签的结束部分
        yield
        # 在 table 列表中添加 HTML 锚标签的结束部分
        table.append("</a>")

    # 使用 contextlib 中的 tag 函数创建一个 HTML 表格的上下文，将生成的标签添加到 table 列表中
    with tag('table'):
        # 在表格中创建一行（tr）
        with tag('tr'):
            # 对于每个表头名称，在表格中创建表头（th）并添加到 table 列表中
            for headname in ["Filename", "Description", "size", "sha256"]:
                with tag("th"):
                    table.append(headname)

        # 遍历 descriptions 字典的键
        for key in descriptions:
            # 调用 get_tarball_name 函数获取 tar 包的名称
            name = get_tarball_name(key, version)
            # 在表格中创建一行（tr）
            with tag('tr'):
                # 在行中创建单元格（td），并创建包含指向 GitHub 上对应文件的链接
                with tag('td'):
                    # 使用 a_href 函数生成的链接标签，并将文件名添加为加粗文本
                    with a_href('https://github.com/sympy/sympy/releases/download/sympy-%s/%s' % (version, name)):
                        with tag('b'):
                            table.append(name)
                # 在行中创建单元格（td），添加描述信息到 table 列表中
                with tag('td'):
                    table.append(descriptions[key].format(**tarball_formatter_dict))
                # 在行中创建单元格（td），添加文件大小到 table 列表中
                with tag('td'):
                    table.append(sizes_dict[name])
                # 在行中创建单元格（td），添加文件的 SHA256 哈希值到 table 列表中
                with tag('td'):
                    table.append(sha256s_dict[name])

    # 将 table 列表中的所有 HTML 元素连接成一个字符串，作为最终输出
    out = ' '.join(table)
    # 返回最终生成的 HTML 表格字符串
    return out
# 创建有序字典 `descriptions`，包含不同版本的描述信息
descriptions = OrderedDict([
    ('source', "The SymPy source installer.",),  # SymPy 源码安装程序的描述
    ('wheel', "A wheel of the package.",),  # 软件包的 wheel 文件描述
    ('html', '''Html documentation. This is the same as
the <a href="https://docs.sympy.org/latest/index.html">online documentation</a>.''',),  # HTML 文档的描述，与在线文档相同
    ('pdf', '''Pdf version of the <a href="https://docs.sympy.org/latest/index.html"> html documentation</a>.''',),  # PDF 版本的 HTML 文档描述
])


def _size(version, print_=True):
    """
    Print the sizes of the release files. Run locally.
    """
    # 使用 `du -h` 命令获取发布文件的大小
    out = run(*(['du', '-h'] + release_files(version)))
    out = [i.split() for i in out]
    # 格式化输出文件大小和文件名
    out = '\n'.join(["%s\t%s" % (i, os.path.split(j)[1]) for i, j in out])
    if print_:
        print(out)
    return out


def _sha256(version, print_=True, local=False):
    if local:
        # 如果是本地操作，使用 `shasum -a 256` 命令计算文件的 SHA-256 哈希值
        out = run(*(['shasum', '-a', '256'] + release_files(version)))
    else:
        raise ValueError('Should not get here...')
        # 否则抛出错误，提示不应该执行到这里
        # out = run(*(['shasum', '-a', '256', '/root/release/*']))
    # 移除输出结果中文件名前面的 `release/` 部分，方便复制到发布说明中
    out = [i.split() for i in out]
    out = '\n'.join(["%s\t%s" % (i, os.path.split(j)[1]) for i, j in out])
    if print_:
        print(out)
    return out


def get_tarball_name(file, version):
    """
    Get the name of a tarball

    file should be one of

    source-orig:       The original name of the source tarball
    source-orig-notar: The name of the untarred directory
    source:            The source tarball (after renaming)
    wheel:             The wheel
    html:              The name of the html zip
    html-nozip:        The name of the html, without ".zip"
    pdf-orig:          The original name of the pdf file
    pdf:               The name of the pdf file (after renaming)
    """
    # 定义文件类型到文件名的映射关系
    doctypename = defaultdict(str, {'html': 'zip', 'pdf': 'pdf'})

    if file in {'source-orig', 'source'}:
        # 根据文件类型选择模板，并用给定的版本号填充
        name = 'sympy-{version}.tar.gz'
    elif file == 'source-orig-notar':
        name = "sympy-{version}"
    elif file in {'html', 'pdf', 'html-nozip'}:
        name = "sympy-docs-{type}-{version}"
        if file == 'html-nozip':
            # 对于 zip 文件，保留原始压缩目录的名称，参考 GitHub issue #7087
            file = 'html'
        else:
            name += ".{extension}"
    elif file == 'pdf-orig':
        name = "sympy-{version}.pdf"
    elif file == 'wheel':
        name = 'sympy-{version}-py3-none-any.whl'
    else:
        raise ValueError(file + " is not a recognized argument")

    # 格式化返回文件名，包括文件类型和扩展名
    ret = name.format(version=version, type=file,
        extension=doctypename[file])
    return ret


def release_files(version):
    """
    Returns the list of local release files
    """
    # 获取特定版本的本地发布文件列表
    paths = glob.glob('release/release-' + version + '/*')
    if not paths:
        raise ValueError("No release files found")
    return paths


# 定义支持的 tarball 名称类型集合
tarball_name_types = {
    'source-orig',
    'source-orig-notar',
    'source',
    'wheel',
    # 定义一个包含字符串元素的集合，用于表示支持的输出格式
    {
        'html',         # HTML 格式输出
        'html-nozip',   # 不经过压缩的 HTML 格式输出
        'pdf-orig',     # 原始 PDF 格式输出
        'pdf',          # PDF 格式输出
    }
# 定义一个类 `_tarball_format`，使其支持类似字典的行为，实现了 Mapping 接口
class _tarball_format(Mapping):

    # 初始化方法，接受一个版本号作为参数
    def __init__(self, version):
        self.version = version

    # 获取指定名称的 tarball 的方法
    def __getitem__(self, name):
        return get_tarball_name(name, self.version)

    # 返回迭代器，用于遍历 tarball_name_types 列表
    def __iter__(self):
        return iter(tarball_name_types)

    # 返回 tarball_name_types 列表的长度
    def __len__(self):
        return len(tarball_name_types)


# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 导入 sys 模块
    import sys
    # 调用 main 函数，并传入命令行参数（去掉第一个参数，即脚本名称）
    main(*sys.argv[1:])
```