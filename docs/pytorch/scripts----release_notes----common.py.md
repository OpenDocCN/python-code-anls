# `.\pytorch\scripts\release_notes\common.py`

```
import json
import locale
import os
import re
import subprocess
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class CategoryGroup:
    name: str
    categories: list


# 定义前端分类列表
frontend_categories = [
    "meta",
    "nn",
    "linalg",
    "cpp",
    "python",
    "complex",
    "vmap",
    "autograd",
    "build",
    "memory_format",
    "foreach",
    "dataloader",
    "sparse",
    "nested tensor",
    "optimizer",
]

# 定义 PyTorch 2.0 版本分类列表
pytorch_2_categories = [
    "dynamo",
    "inductor",
]

# 这些类别将映射到量化分类
quantization = CategoryGroup(
    name="quantization",
    categories=[
        "quantization",
        "AO frontend",
        "AO Pruning",
    ],
)

# Distributed 类别有多个发布说明标签，我们希望将它们映射到一个分类中
distributed = CategoryGroup(
    name="distributed",
    categories=[
        "distributed",
        "distributed (c10d)",
        "distributed (composable)",
        "distributed (ddp)",
        "distributed (fsdp)",
        "distributed (rpc)",
        "distributed (sharded)",
    ],
)

# 定义总类别列表，包含了多个不同的分类
categories = (
    [
        "Uncategorized",
        "lazy",
        "hub",
        "mobile",
        "jit",
        "visualization",
        "onnx",
        "caffe2",
        "amd",
        "rocm",
        "cuda",
        "cpu",
        "cudnn",
        "xla",
        "benchmark",
        "profiler",
        "performance_as_product",
        "package",
        "dispatcher",
        "releng",
        "fx",
        "code_coverage",
        "vulkan",
        "skip",
        "composability",
        # 2.0 release
        "mps",
        "intel",
        "functorch",
        "gnn",
        "distributions",
        "serialization",
    ]
    + [f"{category}_frontend" for category in frontend_categories]  # 添加前端类别到总类别列表中
    + pytorch_2_categories  # 添加 PyTorch 2.0 版本类别到总类别列表中
    + [quantization.name]  # 添加量化分类到总类别列表中
    + [distributed.name]   # 添加分布式分类到总类别列表中
)


# 定义主题列表
topics = [
    "bc breaking",
    "deprecation",
    "new features",
    "improvements",
    "bug fixes",
    "performance",
    "docs",
    "devs",
    "Untopiced",
    "not user facing",
    "security",
]


# 定义命名元组 Features，用于表示特征的结构
Features = namedtuple(
    "Features",
    ["title", "body", "pr_number", "files_changed", "labels", "author", "accepters"],
)


def dict_to_features(dct):
    # 将字典转换为 Features 命名元组
    return Features(
        title=dct["title"],
        body=dct["body"],
        pr_number=dct["pr_number"],
        files_changed=dct["files_changed"],
        labels=dct["labels"],
        author=dct["author"],
        accepters=tuple(dct["accepters"]),
    )


def features_to_dict(features):
    # 将 Features 命名元组转换为字典
    return dict(features._asdict())


def run(command):
    """Returns (return-code, stdout, stderr)"""
    # 执行系统命令，返回执行结果的返回码、标准输出和标准错误
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = output.decode(enc)
    err = err.decode(enc)
    return rc, output.strip(), err.strip()
# 根据提交哈希值获取提交信息的正文部分
def commit_body(commit_hash):
    # 构建执行 Git 命令获取提交信息正文的命令
    cmd = f"git log -n 1 --pretty=format:%b {commit_hash}"
    # 调用 run 函数执行命令，获取返回码、标准输出和标准错误输出
    ret, out, err = run(cmd)
    # 如果返回码为 0，则返回获取到的提交信息正文，否则返回 None
    return out if ret == 0 else None


# 根据提交哈希值获取提交信息的标题部分
def commit_title(commit_hash):
    # 构建执行 Git 命令获取提交信息标题的命令
    cmd = f"git log -n 1 --pretty=format:%s {commit_hash}"
    # 调用 run 函数执行命令，获取返回码、标准输出和标准错误输出
    ret, out, err = run(cmd)
    # 如果返回码为 0，则返回获取到的提交信息标题，否则返回 None
    return out if ret == 0 else None


# 根据提交哈希值获取与之相关的更改文件列表
def commit_files_changed(commit_hash):
    # 构建执行 Git 命令获取提交更改文件列表的命令
    cmd = f"git diff-tree --no-commit-id --name-only -r {commit_hash}"
    # 调用 run 函数执行命令，获取返回码、标准输出和标准错误输出
    ret, out, err = run(cmd)
    # 如果返回码为 0，则将获取到的文件列表按行分割并返回，否则返回 None
    return out.split("\n") if ret == 0 else None


# 解析提交信息正文中的 Pull Request 编号
def parse_pr_number(body, commit_hash, title):
    # 匹配 Pull Request 解析链接的正则表达式
    regex = r"Pull Request resolved: https://github.com/pytorch/pytorch/pull/([0-9]+)"
    # 在提交信息正文中查找匹配的 PR 编号
    matches = re.findall(regex, body)
    # 如果未找到匹配的 PR 编号
    if len(matches) == 0:
        # 如果标题中不包含 "revert" 或 "updating submodules"，则输出警告信息并返回 None
        if "revert" not in title.lower() and "updating submodules" not in title.lower():
            print(f"[{commit_hash}: {title}] Could not parse PR number, ignoring PR")
        return None
    # 如果找到多个匹配的 PR 编号，输出警告信息并使用第一个找到的编号
    if len(matches) > 1:
        print(f"[{commit_hash}: {title}] Got two PR numbers, using the first one")
        return matches[0]
    # 如果找到一个匹配的 PR 编号，直接返回该编号
    return matches[0]


# 获取 GitHub 访问令牌
def get_ghstack_token():
    # 匹配 GitHub 访问令牌的配置模式
    pattern = "github_oauth = (.*)"
    # 打开 ~/.ghstackrc 文件，读取其中的配置信息
    with open(Path("~/.ghstackrc").expanduser(), "r+") as f:
        config = f.read()
    # 在配置信息中查找匹配的 GitHub 访问令牌
    matches = re.findall(pattern, config)
    # 如果未找到匹配的 GitHub 访问令牌，则抛出运行时错误
    if len(matches) == 0:
        raise RuntimeError("Can't find a github oauth token")
    # 返回找到的第一个 GitHub 访问令牌
    return matches[0]


# 获取 GitHub 访问令牌，优先从环境变量 GITHUB_TOKEN 中获取，否则从配置文件中获取
def get_token():
    env_token = os.environ.get("GITHUB_TOKEN")
    # 如果环境变量中存在 GITHUB_TOKEN，输出使用环境变量中的提示信息，并返回该令牌
    if env_token is not None:
        print("using GITHUB_TOKEN from environment variable")
        return env_token
    else:
        # 否则从配置文件中获取 GitHub 访问令牌并返回
        return get_ghstack_token()


# 获取 GitHub API 请求所需的 HTTP 头部信息，包含授权信息
token = get_token()
headers = {"Authorization": f"token {token}"}


# 执行 GraphQL 查询请求，并返回查询结果的 JSON 数据
def run_query(query):
    # 发送 POST 请求到 GitHub GraphQL API，包含查询字符串和授权头部信息
    request = requests.post(
        "https://api.github.com/graphql", json={"query": query}, headers=headers
    )
    # 如果请求返回状态码为 200，则返回 JSON 格式的查询结果
    if request.status_code == 200:
        return request.json()
    # 否则抛出异常，包含请求返回的状态码和错误信息
    else:
        raise Exception(
            f"Query failed to run by returning code of {request.status_code}. {request.json()}"
        )


# GitHub API 请求失败时的错误信息列表
_ERRORS = []
# 最大错误信息长度限制
_MAX_ERROR_LEN = 20


# 根据 Pull Request 编号查询 GitHub 上的相关数据
def github_data(pr_number):
    # 构建 GitHub GraphQL 查询字符串，查询 pytorch 仓库中指定 PR 编号的信息
    query = (
        """
    {
      repository(owner: "pytorch", name: "pytorch") {
        pullRequest(number: %s ) {
          author {
            login
          }
          reviews(last: 5, states: APPROVED) {
            nodes {
              author {
                login
              }
            }
          }
          labels(first: 10) {
            edges {
              node {
                name
              }
            }
          }
        }
      }
    }
    """  # noqa: UP031
        % pr_number
    )
    # 执行 GraphQL 查询请求，并返回查询结果
    query = run_query(query)
    # 如果查询参数中包含"errors"
    if query.get("errors"):
        # 将错误信息添加到全局错误列表_ERRORS中
        global _ERRORS
        _ERRORS.append(query.get("errors"))
        # 如果_ERRORS列表长度小于最大错误长度_MAX_ERROR_LEN
        if len(_ERRORS) < _MAX_ERROR_LEN:
            # 返回空列表、字符串"None"和空元组
            return [], "None", ()
        else:
            # 如果_ERRORS列表长度超过或等于_MAX_ERROR_LEN，抛出异常
            raise Exception(
                f"Got {_MAX_ERROR_LEN} errors: {_ERRORS}, please check if"
                " there is something wrong"
            )

    # 获取查询结果中的标签列表
    edges = query["data"]["repository"]["pullRequest"]["labels"]["edges"]
    # 提取标签节点中的名称，组成列表labels
    labels = [edge["node"]["name"] for edge in edges]
    # 获取提交者的登录名
    author = query["data"]["repository"]["pullRequest"]["author"]["login"]
    # 获取评论节点列表
    nodes = query["data"]["repository"]["pullRequest"]["reviews"]["nodes"]

    # 使用集合去重，获取所有接受者的登录名
    accepters = {node["author"]["login"] for node in nodes}
    # 将接受者登录名排序后转换为元组
    accepters = tuple(sorted(accepters))

    # 返回标签列表、提交者登录名和接受者登录名元组
    return labels, author, accepters
# 根据提交哈希获取与该提交相关的特征信息，并返回一个 Features 对象
def get_features(commit_hash):
    # 获取提交的标题、正文和修改的文件列表
    title, body, files_changed = (
        commit_title(commit_hash),
        commit_body(commit_hash),
        commit_files_changed(commit_hash),
    )
    # 从提交正文中解析出关联的 PR 编号
    pr_number = parse_pr_number(body, commit_hash, title)
    # 初始化标签、作者和接受者列表
    labels = []
    author = ""
    accepters = tuple()
    # 如果存在关联的 PR 编号，则获取 GitHub 数据：标签、作者和接受者列表
    if pr_number is not None:
        labels, author, accepters = github_data(pr_number)
    # 创建 Features 对象，包含标题、正文、PR 编号、修改的文件列表、标签、作者和接受者列表
    result = Features(title, body, pr_number, files_changed, labels, author, accepters)
    return result


# 全局变量，用于缓存提交数据
_commit_data_cache = None


# 获取提交数据的缓存对象，如果不存在则初始化并返回缓存对象
def get_commit_data_cache(path="results/data.json"):
    global _commit_data_cache
    if _commit_data_cache is None:
        _commit_data_cache = _CommitDataCache(path)
    return _commit_data_cache


# 私有类 _CommitDataCache，用于管理提交数据的缓存
class _CommitDataCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        # 如果指定路径存在，则从磁盘读取数据到缓存
        if os.path.exists(path):
            self.data = self.read_from_disk()
        else:
            # 否则，创建路径中缺失的目录
            os.makedirs(Path(path).parent, exist_ok=True)

    # 根据提交哈希获取缓存中的数据，如果不存在则获取新数据并缓存
    def get(self, commit):
        if commit not in self.data.keys():
            # 获取新的特征数据并缓存
            self.data[commit] = get_features(commit)
            # 将更新后的数据写入磁盘
            self.write_to_disk()
        return self.data[commit]

    # 从磁盘中读取数据到缓存
    def read_from_disk(self):
        with open(self.path) as f:
            data = json.load(f)
            # 将 JSON 格式的数据转换为 Features 对象
            data = {commit: dict_to_features(dct) for commit, dct in data.items()}
        return data

    # 将缓存中的数据写入磁盘
    def write_to_disk(self):
        # 将 Features 对象转换为字典格式并写入 JSON 文件
        data = {commit: features._asdict() for commit, features in self.data.items()}
        with open(self.path, "w") as f:
            json.dump(data, f)
```