# `.\pytorch\tools\stats\upload_stats_lib.py`

```py
# 从未来版本中导入注释类型，这样代码可以与较旧的 Python 版本兼容
from __future__ import annotations

# 导入 gzip、io、json、os、zipfile 等标准库
import gzip
import io
import json
import os
import zipfile

# 从 pathlib 中导入 Path 类型，用于处理文件路径
from pathlib import Path

# 导入 Any 类型，用于表示任意类型的对象
from typing import Any

# 导入 boto3、requests、rockset 等外部库
import boto3  # type: ignore[import]
import requests
import rockset  # type: ignore[import]

# PyTorch 仓库的 GitHub API 地址
PYTORCH_REPO = "https://api.github.com/repos/pytorch/pytorch"

# 使用 boto3 创建 S3 资源对象
S3_RESOURCE = boto3.resource("s3")

# 在持续集成环境中，非禁用模式下最大重试次数
MAX_RETRY_IN_NON_DISABLED_MODE = 3 * 3

# Rockset 单个请求中文档的最大数量限制
BATCH_SIZE = 5000


def _get_request_headers() -> dict[str, str]:
    # 返回 GitHub API 请求所需的头部信息，包括接受的数据类型和授权令牌
    return {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + os.environ["GITHUB_TOKEN"],
    }


def _get_artifact_urls(prefix: str, workflow_run_id: int) -> dict[Path, str]:
    """获取所有名称中带有 'test-report' 的工作流程工件的 URL。"""
    # 发送 HTTP GET 请求获取指定工作流程运行中的所有工件信息
    response = requests.get(
        f"{PYTORCH_REPO}/actions/runs/{workflow_run_id}/artifacts?per_page=100",
        headers=_get_request_headers(),
    )
    # 解析响应内容中的工件列表
    artifacts = response.json()["artifacts"]

    # 处理分页请求，直到获取所有工件信息
    while "next" in response.links.keys():
        response = requests.get(
            response.links["next"]["url"], headers=_get_request_headers()
        )
        artifacts.extend(response.json()["artifacts"])

    # 创建字典存储符合前缀要求的工件名称和其对应的下载 URL
    artifact_urls = {}
    for artifact in artifacts:
        if artifact["name"].startswith(prefix):
            artifact_urls[Path(artifact["name"])] = artifact["archive_download_url"]
    return artifact_urls


def _download_artifact(
    artifact_name: Path, artifact_url: str, workflow_run_attempt: int
) -> Path:
    # [工件运行尝试]
    # 所有工件共享一个命名空间。但是，可以重新运行工作流程并生成新的工件集合。
    # 为了避免名称冲突，在工件名称中添加 '-runattempt1<run #>-' 以表示运行尝试次数。
    #
    # 此代码从工件名称中解析出运行尝试号。如果与命令行指定的不匹配，则跳过该工件。
    atoms = str(artifact_name).split("-")
    for atom in atoms:
        if atom.startswith("runattempt"):
            found_run_attempt = int(atom[len("runattempt") :])
            if workflow_run_attempt != found_run_attempt:
                print(
                    f"Skipping {artifact_name} as it is an invalid run attempt. "
                    f"Expected {workflow_run_attempt}, found {found_run_attempt}."
                )

    # 打印正在下载的工件名称
    print(f"Downloading {artifact_name}")

    # 发送 HTTP GET 请求下载工件内容，并保存到本地文件
    response = requests.get(artifact_url, headers=_get_request_headers())
    with open(artifact_name, "wb") as f:
        f.write(response.content)
    return artifact_name


def download_s3_artifacts(
    prefix: str, workflow_run_id: int, workflow_run_attempt: int
) -> list[Path]:
    # 创建 S3 存储桶资源对象
    bucket = S3_RESOURCE.Bucket("gha-artifacts")

    # 使用前缀、工作流程运行 ID 和运行尝试 ID 过滤 S3 存储桶中的对象
    objs = bucket.objects.filter(
        Prefix=f"pytorch/pytorch/{workflow_run_id}/{workflow_run_attempt}/artifact/{prefix}"
    )

    # 注意：此处缺少完整的函数实现，请确保实现 download_s3_artifacts 函数的完整性和正确性。
    # 初始化一个布尔变量，用来标记是否找到了至少一个对象
    found_one = False
    # 初始化一个空列表，用来存储下载的文件路径
    paths = []
    # 遍历传入的对象列表
    for obj in objs:
        # 标记为找到至少一个对象
        found_one = True
        # 从对象的键中提取文件名，并创建路径对象
        p = Path(Path(obj.key).name)
        # 打印下载信息，包含文件名
        print(f"Downloading {p}")
        # 打开文件 p，以二进制写入模式，准备写入对象内容
        with open(p, "wb") as f:
            # 从对象的内容中读取二进制数据，并写入文件 f
            f.write(obj.get()["Body"].read())
        # 将下载完成的文件路径添加到路径列表中
        paths.append(p)

    # 如果没有找到任何对象
    if not found_one:
        # 打印警告信息，提示未在 S3 中找到任何测试报告
        print(
            "::warning title=s3 artifacts not found::"
            "Didn't find any test reports in s3, there might be a bug!"
        )
    # 返回存储了所有下载文件路径的列表
    return paths
# 下载 GitHub Actions 构建产物并返回路径列表
def download_gha_artifacts(
    prefix: str, workflow_run_id: int, workflow_run_attempt: int
) -> list[Path]:
    # 获取构建产物的下载链接字典
    artifact_urls = _get_artifact_urls(prefix, workflow_run_id)
    paths = []
    # 遍历下载链接字典，下载每个构建产物并添加路径到列表中
    for name, url in artifact_urls.items():
        paths.append(_download_artifact(Path(name), url, workflow_run_attempt))
    return paths


# 将文档批量上传到 Rockset
def upload_to_rockset(
    collection: str,
    docs: list[Any],
    workspace: str = "commons",
    client: Any = None,
) -> None:
    if not client:
        # 若未提供客户端，则创建一个连接到 Rockset 的客户端
        client = rockset.RocksetClient(
            host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
        )

    index = 0
    while index < len(docs):
        from_index = index
        to_index = min(from_index + BATCH_SIZE, len(docs))
        # 打印当前批次的文档数量并上传到 Rockset
        print(f"Writing {to_index - from_index} documents to Rockset")

        client.Documents.add_documents(
            collection=collection,
            data=docs[from_index:to_index],
            workspace=workspace,
        )
        index += BATCH_SIZE

    # 完成上传后打印信息
    print("Done!")


# 将文档批量写入到 S3 存储桶
def upload_to_s3(
    bucket_name: str,
    key: str,
    docs: list[dict[str, Any]],
) -> None:
    # 打印将要写入的文档数量
    print(f"Writing {len(docs)} documents to S3")
    body = io.StringIO()
    # 将每个文档以 JSON 格式写入 StringIO 对象
    for doc in docs:
        json.dump(doc, body)
        body.write("\n")

    # 使用 gzip 压缩后上传至 S3 对象
    S3_RESOURCE.Object(
        f"{bucket_name}",
        f"{key}",
    ).put(
        Body=gzip.compress(body.getvalue().encode()),
        ContentEncoding="gzip",
        ContentType="application/json",
    )
    # 完成上传后打印信息
    print("Done!")


# 从 S3 中读取文档并返回为列表
def read_from_s3(
    bucket_name: str,
    key: str,
) -> list[dict[str, Any]]:
    # 打印读取的 S3 对象信息
    print(f"Reading from s3://{bucket_name}/{key}")
    # 读取并解压缩从 S3 获取的数据，解析为 JSON 列表
    body = (
        S3_RESOURCE.Object(
            f"{bucket_name}",
            f"{key}",
        )
        .get()["Body"]
        .read()
    )
    results = gzip.decompress(body).decode().split("\n")
    return [json.loads(result) for result in results if result]


# 将工作流统计信息上传到指定 S3 存储桶路径
def upload_workflow_stats_to_s3(
    workflow_run_id: int,
    workflow_run_attempt: int,
    collection: str,
    docs: list[dict[str, Any]],
) -> None:
    bucket_name = "ossci-raw-job-status"
    key = f"{collection}/{workflow_run_id}/{workflow_run_attempt}"
    # 调用上传至 S3 的函数
    upload_to_s3(bucket_name, key, docs)


# 上传本地文件至 S3 存储桶
def upload_file_to_s3(
    file_name: str,
    bucket: str,
    key: str,
) -> None:
    """
    Upload a local file to S3
    """
    # 打印上传文件的信息
    print(f"Upload {file_name} to s3://{bucket}/{key}")
    # 使用 boto3 客户端上传文件至 S3
    boto3.client("s3").upload_file(
        file_name,
        bucket,
        key,
    )


# 解压提供的 zip 文件到同名目录
def unzip(p: Path) -> None:
    """Unzip the provided zipfile to a similarly-named directory.

    Returns None if `p` is not a zipfile.

    Looks like: /tmp/test-reports.zip -> /tmp/unzipped-test-reports/
    """
    assert p.is_file()
    # 创建解压后的目标目录，并打印解压信息
    unzipped_dir = p.with_name("unzipped-" + p.stem)
    print(f"Extracting {p} to {unzipped_dir}")

    # 使用 zipfile 库解压文件
    with zipfile.ZipFile(p, "r") as zip:
        zip.extractall(unzipped_dir)


# 检查是否存在需要禁用的测试用例
def is_rerun_disabled_tests(tests: dict[str, dict[str, int]]) -> bool:
    """
    Check if there are disabled tests that need rerun
    """
    # 检查测试报告是否来自 rerun_disabled_tests 工作流，该工作流中每个测试运行多次
    """
    Check if the test report is coming from rerun_disabled_tests workflow where
    each test is run multiple times
    """
    # 对于 tests 字典中的每个测试 t，检查其 num_green 和 num_red 属性之和是否大于 MAX_RETRY_IN_NON_DISABLED_MODE
    return all(
        t.get("num_green", 0) + t.get("num_red", 0) > MAX_RETRY_IN_NON_DISABLED_MODE
        for t in tests.values()
    )
# [Job id in artifacts]
# 从报告路径中获取作业 ID。在我们的 GitHub Actions 工作流中，我们将作业 ID 添加到报告名的末尾，
# 因此 `report` 的形式通常是这样的：
#     unzipped-test-reports-foo_5596745227/test/test-reports/foo/TEST-foo.xml
# 我们想要从中提取出 `5596745227`。
def get_job_id(report: Path) -> int | None:
    try:
        # 从路径的第一部分中提取作业 ID，该部分通过下划线分隔，取最后一部分并转换为整数返回
        return int(report.parts[0].rpartition("_")[2])
    except ValueError:
        # 如果无法转换为整数（比如提取的部分不是有效的数字），返回 None
        return None
```