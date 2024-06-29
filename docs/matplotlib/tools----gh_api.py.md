# `D:\src\scipysrc\matplotlib\tools\gh_api.py`

```
"""Functions for GitHub API requests."""

# 导入所需的模块和库
import getpass  # 导入用于安全地获取密码的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
import re  # 导入正则表达式模块
import sys  # 导入与 Python 解释器交互的模块

import requests  # 导入处理 HTTP 请求的库

try:
    import requests_cache  # 尝试导入请求缓存模块
except ImportError:
    # 若导入失败，则打印错误信息到标准错误流
    print("no cache", file=sys.stderr)
else:
    # 若导入成功，则安装请求缓存，缓存名称为"gh_api"，有效期3600秒
    requests_cache.install_cache("gh_api", expire_after=3600)

# Keyring 存储密码需要一个“用户名”，这里使用一个虚拟的用户名
fake_username = 'ipython_tools'

class Obj(dict):
    """Dictionary with attribute access to names."""
    
    # 允许通过属性访问字典中的键值
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    # 允许设置字典的属性
    def __setattr__(self, name, val):
        self[name] = val

token = None
def get_auth_token():
    global token

    # 如果已经有缓存的 token，则直接返回
    if token is not None:
        return token

    try:
        # 尝试从用户的家目录下的".ghoauth"文件中读取 token
        with open(os.path.join(os.path.expanduser('~'), '.ghoauth')) as f:
            token, = f  # 从文件中读取 token
            return token
    except Exception:
        pass

    import keyring
    # 使用 Keyring 获取 GitHub 的 token
    token = keyring.get_password('github', fake_username)
    if token is not None:
        return token

    # 如果 Keyring 中没有 token，则需要用户输入 GitHub 用户名和密码
    print("Please enter your github username and password. These are not "
           "stored, only used to get an oAuth token. You can revoke this at "
           "any time on GitHub.")
    user = input("Username: ")  # 获取用户输入的用户名
    pw = getpass.getpass("Password: ")  # 获取用户输入的密码

    auth_request = {
      "scopes": [
        "public_repo",
        "gist"
      ],
      "note": "IPython tools",
      "note_url": "https://github.com/ipython/ipython/tree/master/tools",
    }
    # 发送 POST 请求到 GitHub API 获取授权 token
    response = requests.post('https://api.github.com/authorizations',
                            auth=(user, pw), data=json.dumps(auth_request))
    response.raise_for_status()  # 如果请求失败，则抛出异常
    token = json.loads(response.text)['token']  # 从返回的 JSON 中提取 token
    keyring.set_password('github', fake_username, token)  # 将 token 存入 Keyring 中
    return token

def make_auth_header():
    # 创建包含授权头部信息的字典
    return {'Authorization': 'token ' + get_auth_token().replace("\n","")}

def post_issue_comment(project, num, body):
    # 发布评论到 GitHub Issue 上
    url = f'https://api.github.com/repos/{project}/issues/{num}/comments'
    payload = json.dumps({'body': body})  # 构造评论内容的 JSON
    requests.post(url, data=payload, headers=make_auth_header())  # 发送 POST 请求

def post_gist(content, description='', filename='file', auth=False):
    """Post some text to a Gist, and return the URL."""
    post_data = json.dumps({
      "description": description,
      "public": True,
      "files": {
        filename: {
          "content": content
        }
      }
    }).encode('utf-8')

    headers = make_auth_header() if auth else {}  # 根据 auth 参数决定是否需要授权头部信息
    response = requests.post("https://api.github.com/gists", data=post_data, headers=headers)  # 发送 POST 请求到 GitHub Gist API
    response.raise_for_status()  # 如果请求失败，则抛出异常
    response_data = json.loads(response.text)  # 解析返回的 JSON 数据
    return response_data['html_url']  # 返回创建的 Gist 的 URL

def get_pull_request(project, num, auth=False):
    """Return the pull request info for a given PR number."""
    url = f"https://api.github.com/repos/{project}/pulls/{num}"
    if auth:
        header = make_auth_header()  # 如果需要授权，则获取授权头部信息
    else:
        header = None
    # 打印正在获取的 URL 到标准错误输出
    print("fetching %s" % url, file=sys.stderr)
    # 发送 GET 请求到指定的 URL，并使用给定的头部信息
    response = requests.get(url, headers=header)
    # 如果响应状态码不是成功状态，会抛出对应的 HTTPError 异常
    response.raise_for_status()
    # 解析 JSON 格式的响应文本，使用自定义的对象钩子 Obj 来处理对象转换
    return json.loads(response.text, object_hook=Obj)
# 获取指定项目中某个拉取请求中的文件列表
def get_pull_request_files(project, num, auth=False):
    """get list of files in a pull request"""
    # 构建访问 GitHub API 的 URL
    url = f"https://api.github.com/repos/{project}/pulls/{num}/files"
    # 根据是否需要认证，决定是否添加授权头部信息
    if auth:
        header = make_auth_header()
    else:
        header = None
    # 调用 API 请求函数，并返回结果
    return get_paged_request(url, headers=header)

# 编译正则表达式，用于查找 HTML 标签元素
element_pat = re.compile(r'<(.+?)>')
# 编译正则表达式，用于查找关系链接的关系类型
rel_pat = re.compile(r'rel=[\'"](\w+)[\'"]')

# 获取 GitHub API 的分页请求结果并合并返回
def get_paged_request(url, headers=None, **params):
    """get a full list, handling APIv3's paging"""
    # 初始化空列表，用于存储所有请求结果
    results = []
    # 设置每页默认返回条目数为 100 条
    params.setdefault("per_page", 100)
    # 循环获取所有分页数据
    while True:
        # 根据 URL 是否包含查询参数打印不同的信息到标准错误输出
        if '?' in url:
            params = None
            print(f"fetching {url}", file=sys.stderr)
        else:
            print(f"fetching {url} with {params}", file=sys.stderr)
        # 发起 GET 请求获取数据，并根据 headers 和 params 发起请求
        response = requests.get(url, headers=headers, params=params)
        # 检查请求是否成功，否则抛出异常
        response.raise_for_status()
        # 将获取的 JSON 数据追加到结果列表中
        results.extend(response.json())
        # 如果响应中包含 'next' 链接，则更新 URL 继续下一页数据获取
        if 'next' in response.links:
            url = response.links['next']['url']
        else:
            break
    # 返回所有获取的结果数据
    return results

# 获取指定项目的已关闭的拉取请求列表
def get_pulls_list(project, auth=False, **params):
    """get pull request list"""
    # 设置默认参数状态为 'closed'
    params.setdefault("state", "closed")
    # 构建访问 GitHub API 的 URL
    url = f"https://api.github.com/repos/{project}/pulls"
    # 根据是否需要认证，决定是否添加授权头部信息
    if auth:
        headers = make_auth_header()
    else:
        headers = None
    # 获取并返回分页请求结果
    pages = get_paged_request(url, headers=headers, **params)
    return pages

# 获取指定项目的已关闭的问题列表
def get_issues_list(project, auth=False, **params):
    """get issues list"""
    # 设置默认参数状态为 'closed'
    params.setdefault("state", "closed")
    # 构建访问 GitHub API 的 URL
    url = f"https://api.github.com/repos/{project}/issues"
    # 根据是否需要认证，决定是否添加授权头部信息
    if auth:
        headers = make_auth_header()
    else:
        headers = None
    # 获取并返回分页请求结果
    pages = get_paged_request(url, headers=headers, **params)
    return pages

# 获取指定项目的里程碑列表
def get_milestones(project, auth=False, **params):
    # 设置默认参数状态为 'all'
    params.setdefault('state', 'all')
    # 构建访问 GitHub API 的 URL
    url = f"https://api.github.com/repos/{project}/milestones"
    # 根据是否需要认证，决定是否添加授权头部信息
    if auth:
        headers = make_auth_header()
    else:
        headers = None
    # 获取并返回分页请求结果
    milestones = get_paged_request(url, headers=headers, **params)
    return milestones

# 获取指定项目中指定里程碑的编号
def get_milestone_id(project, milestone, auth=False, **params):
    # 获取指定项目的所有里程碑列表
    milestones = get_milestones(project, auth=auth, **params)
    # 遍历里程碑列表，查找匹配指定里程碑名称的编号
    for mstone in milestones:
        if mstone['title'] == milestone:
            return mstone['number']
    # 若未找到匹配的里程碑名称，抛出值错误异常
    raise ValueError("milestone %s not found" % milestone)

# 判断给定的问题是否为拉取请求，返回布尔值
def is_pull_request(issue):
    """Return True if the given issue is a pull request."""
    return bool(issue.get('pull_request', {}).get('html_url', None))

# 获取指定拉取请求的作者列表
def get_authors(pr):
    # 打印获取指定拉取请求作者信息的调试消息
    print("getting authors for #%i" % pr['number'], file=sys.stderr)
    # 获取授权头部信息
    h = make_auth_header()
    # 发起 GET 请求获取拉取请求的提交信息
    r = requests.get(pr['commits_url'], headers=h)
    # 检查请求是否成功，否则抛出异常
    r.raise_for_status()
    # 将获取的提交信息转换为 JSON 格式
    commits = r.json()
    # 初始化作者列表
    authors = []
    # 遍历提交信息，获取每个提交的作者信息，并添加到作者列表中
    for commit in commits:
        author = commit['commit']['author']
        authors.append(f"{author['name']} <{author['email']}>")
    # 返回作者列表
    return authors

# encode_multipart_formdata is from urllib3.filepost
# 定义函数 iter_fields，用于迭代处理字段，确保符合 S3 所需的键顺序

def iter_fields(fields):
    # 复制字段字典，以便安全地修改和迭代
    fields = fields.copy()
    # 按照 S3 要求的顺序，依次迭代处理字段
    for key in [
            'key', 'acl', 'Filename', 'success_action_status',
            'AWSAccessKeyId', 'Policy', 'Signature', 'Content-Type', 'file']:
        # 弹出指定键的值，确保这些键在迭代过程中被处理
        yield key, fields.pop(key)
    # 继续迭代处理剩余的字段
    yield from fields.items()

# 定义函数 encode_multipart_formdata，用于将字段编码为 multipart/form-data 格式的数据

def encode_multipart_formdata(fields, boundary=None):
    """
    Encode a dictionary of ``fields`` using the multipart/form-data mime format.

    :param fields:
        Dictionary of fields or list of (key, value) field tuples.  The key is
        treated as the field name, and the value as the body of the form-data
        bytes. If the value is a tuple of two elements, then the first element
        is treated as the filename of the form-data section.

        Field names and filenames must be str.

    :param boundary:
        If not specified, then a random boundary will be generated using
        :func:`mimetools.choose_boundary`.
    """
    # 导入所需的模块和函数
    from io import BytesIO
    from requests.packages.urllib3.filepost import (
        choose_boundary, writer, b, get_content_type
    )
    # 创建字节流对象
    body = BytesIO()
    # 如果未指定 boundary，则随机生成一个
    if boundary is None:
        boundary = choose_boundary()

    # 迭代处理每个字段和对应的值
    for fieldname, value in iter_fields(fields):
        body.write(b('--%s\r\n' % (boundary)))

        if isinstance(value, tuple):
            filename, data = value
            # 写入带文件名的内容描述部分的头部
            writer(body).write('Content-Disposition: form-data; name="%s"; '
                               'filename="%s"\r\n' % (fieldname, filename))
            # 写入文件类型的内容头部
            body.write(b('Content-Type: %s\r\n\r\n' %
                       (get_content_type(filename))))
        else:
            data = value
            # 写入普通字段的内容描述部分的头部
            writer(body).write('Content-Disposition: form-data; name="%s"\r\n'
                               % (fieldname))
            # 写入纯文本类型的内容头部
            body.write(b'Content-Type: text/plain\r\n\r\n')

        # 处理数据部分
        if isinstance(data, int):
            data = str(data)  # 兼容处理整数类型的数据
        if isinstance(data, str):
            writer(body).write(data)
        else:
            body.write(data)

        # 写入内容结束的换行
        body.write(b'\r\n')

    # 写入整个 multipart/form-data 的结束符
    body.write(b('--%s--\r\n' % (boundary)))

    # 设置内容类型头部
    content_type = b('multipart/form-data; boundary=%s' % boundary)

    # 返回编码后的数据和内容类型头部
    return body.getvalue(), content_type


# 定义函数 post_download，用于上传文件到 GitHub 的下载区域

def post_download(project, filename, name=None, description=""):
    """Upload a file to the GitHub downloads area"""
    # 如果未提供文件名，则使用文件路径中的基本文件名
    if name is None:
        name = os.path.basename(filename)
    # 打开文件并读取其内容
    with open(filename, 'rb') as f:
        filedata = f.read()

    # 构造 GitHub API 的下载 URL
    url = f"https://api.github.com/repos/{project}/downloads"

    # 构造包含文件信息的 JSON 数据
    payload = json.dumps(dict(name=name, size=len(filedata),
                    description=description))
    
    # 发送 POST 请求到 GitHub API，使用授权头部
    response = requests.post(url, data=payload, headers=make_auth_header())
    # 检查响应状态，抛出异常以处理错误情况
    response.raise_for_status()
    # 解析响应内容为 JSON 格式
    reply = json.loads(response.content)
    # 获取 S3 存储的 URL
    s3_url = reply['s3_url']
    # 构建包含 S3 所需字段的字典
    fields = dict(
        key=reply['path'],                     # S3 上文件的路径
        acl=reply['acl'],                      # 访问控制列表（ACL）
        success_action_status=201,             # 指定成功响应状态码为 201
        Filename=reply['name'],                # 文件名
        AWSAccessKeyId=reply['accesskeyid'],   # AWS 访问密钥 ID
        Policy=reply['policy'],                # 访问策略
        Signature=reply['signature'],          # 签名信息
        file=(reply['name'], filedata),        # 文件数据，元组包含文件名和文件内容
    )
    # 添加文件的 MIME 类型到字段字典
    fields['Content-Type'] = reply['mime_type']
    # 编码表单数据，并获取编码后的数据和 Content-Type
    data, content_type = encode_multipart_formdata(fields)
    # 发送 POST 请求到 S3 URL，使用编码后的数据和指定的 Content-Type
    s3r = requests.post(s3_url, data=data, headers={'Content-Type': content_type})
    # 返回 S3 的响应对象
    return s3r
```