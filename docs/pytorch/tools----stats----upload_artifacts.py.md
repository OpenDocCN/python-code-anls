# `.\pytorch\tools\stats\upload_artifacts.py`

```py
# 导入必要的模块
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式操作的支持
from tempfile import TemporaryDirectory  # 用于创建临时目录

# 导入自定义的函数
from tools.stats.upload_stats_lib import download_gha_artifacts, upload_file_to_s3

# 定义常量：需要下载的 GHA（GitHub Actions）工件的列表
ARTIFACTS = [
    "sccache-stats",
    "test-jsons",
    "test-reports",
    "usage-log",
]

# 定义 S3 存储桶的名称
BUCKET_NAME = "gha-artifacts"

# 定义文件名的正则表达式，用于匹配并替换文件名中的 runattempt 后缀
FILENAME_REGEX = r"-runattempt\d+"


# 定义函数：从 GitHub Actions 下载工件并上传到 S3
def get_artifacts(repo: str, workflow_run_id: int, workflow_run_attempt: int) -> None:
    # 使用临时目录来进行操作
    with TemporaryDirectory() as temp_dir:
        # 打印临时目录的路径
        print("Using temporary directory:", temp_dir)
        # 切换工作目录到临时目录
        os.chdir(temp_dir)

        # 遍历需要下载的每一个工件
        for artifact in ARTIFACTS:
            # 下载指定工件的所有路径列表
            artifact_paths = download_gha_artifacts(
                artifact, workflow_run_id, workflow_run_attempt
            )

            # 遍历每个下载的工件路径
            for artifact_path in artifact_paths:
                # GHA 工件的命名约定为：NAME-runattempt${{ github.run_attempt }}-SUFFIX.zip
                # 我们希望去除 run_attempt 部分，以符合 S3 上的命名约定
                s3_filename = re.sub(FILENAME_REGEX, "", artifact_path.name)
                # 将文件上传到 S3
                upload_file_to_s3(
                    file_name=str(artifact_path.resolve()),  # 获取文件的绝对路径并转为字符串
                    bucket=BUCKET_NAME,  # 指定 S3 存储桶名称
                    key=f"{repo}/{workflow_run_id}/{workflow_run_attempt}/artifact/{s3_filename}",  # S3 上的目标路径
                )


# 如果该脚本被直接执行
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Upload test artifacts from GHA to S3")
    
    # 添加必需的命令行参数
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="which GitHub repo this workflow run belongs to",
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 get_artifacts 函数，传入解析后的参数
    get_artifacts(args.repo, args.workflow_run_id, args.workflow_run_attempt)
```