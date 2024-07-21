# `.\pytorch\ios\TestApp\run_on_aws_devicefarm.py`

```
# 指定 Python 解释器位置为当前环境下的 Python 3
#!/usr/bin/env python3

# 导入日期时间处理模块
import datetime
# 导入操作系统相关功能的模块
import os
# 导入生成随机数和字符串的模块
import random
import string
# 导入系统相关的模块
import sys
# 导入时间相关的模块
import time
# 导入警告相关的模块
import warnings
# 导入用于类型提示的模块
from typing import Any

# 导入 AWS 的 Python SDK
import boto3
# 导入处理 HTTP 请求的模块
import requests

# 定义轮询延迟时间为 5 秒
POLLING_DELAY_IN_SECOND = 5
# 定义最大上传等待时间为 600 秒
MAX_UPLOAD_WAIT_IN_SECOND = 600

# 定义默认的 AWS 设备池 ARN
# NB: 这是 AWS 中精心挑选的顶级设备。我们可以根据需要创建自己的设备池。
DEFAULT_DEVICE_POOL_ARN = (
    "arn:aws:devicefarm:us-west-2::devicepool:082d10e5-d7d7-48a5-ba5c-b33d66efa1f5"
)


def parse_args() -> Any:
    # 导入参数解析模块
    from argparse import ArgumentParser

    # 创建参数解析器实例，描述为“在 AWS 设备农场上运行 iOS 测试”
    parser = ArgumentParser("Run iOS tests on AWS Device Farm")
    # 添加项目 ARN 参数，类型为字符串，必填，帮助信息为“在 AWS 上的项目 ARN”
    parser.add_argument(
        "--project-arn", type=str, required=True, help="the ARN of the project on AWS"
    )
    # 添加应用文件参数，类型为字符串，必填，帮助信息为“iOS ipa 应用程序存档”
    parser.add_argument(
        "--app-file", type=str, required=True, help="the iOS ipa app archive"
    )
    # 添加 XCTest 测试套件文件参数，类型为字符串，必填，帮助信息为“要运行的 XCTest 测试套件”
    parser.add_argument(
        "--xctest-file",
        type=str,
        required=True,
        help="the XCTest suite to run",
    )
    # 添加名称前缀参数，类型为字符串，必填，帮助信息为“此测试运行的名称前缀”
    parser.add_argument(
        "--name-prefix",
        type=str,
        required=True,
        help="the name prefix of this test run",
    )
    # 添加设备池 ARN 参数，类型为字符串，默认为预定义的设备池 ARN，帮助信息为“要测试的设备池的名称”
    parser.add_argument(
        "--device-pool-arn",
        type=str,
        default=DEFAULT_DEVICE_POOL_ARN,
        help="the name of the device pool to test on",
    )

    # 解析并返回命令行参数
    return parser.parse_args()


def upload_file(
    client: Any,
    project_arn: str,
    prefix: str,
    filename: str,
    filetype: str,
    mime: str = "application/octet-stream",
):
    """
    Upload the app file and XCTest suite to AWS
    """
    # 创建上传请求，指定项目 ARN、上传文件名、文件类型和内容类型
    r = client.create_upload(
        projectArn=project_arn,
        name=f"{prefix}_{os.path.basename(filename)}",
        type=filetype,
        contentType=mime,
    )
    # 获取上传结果中的文件名、ARN 和 URL
    upload_name = r["upload"]["name"]
    upload_arn = r["upload"]["arn"]
    upload_url = r["upload"]["url"]

    # 打开本地文件，准备上传到 AWS 设备农场
    with open(filename, "rb") as file_stream:
        # 打印上传信息
        print(f"Uploading {filename} to Device Farm as {upload_name}...")
        # 发送 HTTP PUT 请求进行文件上传，设置请求头为指定的 MIME 类型
        r = requests.put(upload_url, data=file_stream, headers={"content-type": mime})
        # 检查上传是否成功
        if not r.ok:
            raise Exception(f"Couldn't upload {filename}: {r.reason}")  # noqa: TRY002

    # 记录上传开始时间
    start_time = datetime.datetime.now()
    # 轮询 AWS，直到上传的文件准备就绪
    while True:
        # 计算已等待时间
        waiting_time = datetime.datetime.now() - start_time
        # 如果等待时间超过最大上传等待时间，则抛出异常
        if waiting_time > datetime.timedelta(seconds=MAX_UPLOAD_WAIT_IN_SECOND):
            raise Exception(  # noqa: TRY002
                f"Uploading {filename} is taking longer than {MAX_UPLOAD_WAIT_IN_SECOND} seconds, terminating..."
            )

        # 查询上传状态
        r = client.get_upload(arn=upload_arn)
        status = r["upload"].get("status", "")

        # 打印当前文件状态和已等待时间
        print(f"{filename} is in state {status} after {waiting_time}")

        # 如果上传失败，则抛出异常
        if status == "FAILED":
            raise Exception(f"Couldn't upload {filename}: {r}")  # noqa: TRY002
        # 如果上传成功，则跳出循环
        if status == "SUCCEEDED":
            break

        # 等待指定时间后再次轮询
        time.sleep(POLLING_DELAY_IN_SECOND)

    # 返回上传文件的 ARN
    return upload_arn


def main() -> None:
    # 解析命令行参数
    args = parse_args()
    # 创建 boto3 客户端对象，用于与 AWS Device Farm 服务交互
    client = boto3.client("devicefarm")
    
    # 生成一个唯一的前缀，结合名称前缀、当前日期和随机生成的8位字母序列
    unique_prefix = f"{args.name_prefix}-{datetime.date.today().isoformat()}-{''.join(random.sample(string.ascii_letters, 8))}"

    # 上传测试应用程序文件到 AWS Device Farm，并获取返回的文件 ARN
    appfile_arn = upload_file(
        client=client,
        project_arn=args.project_arn,
        prefix=unique_prefix,
        filename=args.app_file,
        filetype="IOS_APP",
    )
    print(f"Uploaded app: {appfile_arn}")
    
    # 上传 XCTest 测试套件文件到 AWS Device Farm，并获取返回的文件 ARN
    xctest_arn = upload_file(
        client=client,
        project_arn=args.project_arn,
        prefix=unique_prefix,
        filename=args.xctest_file,
        filetype="XCTEST_TEST_PACKAGE",
    )
    print(f"Uploaded XCTest: {xctest_arn}")

    # 调度运行测试，返回运行的详细信息
    r = client.schedule_run(
        projectArn=args.project_arn,
        name=unique_prefix,
        appArn=appfile_arn,
        devicePoolArn=args.device_pool_arn,
        test={"type": "XCTEST", "testPackageArn": xctest_arn},
    )
    run_arn = r["run"]["arn"]

    # 记录测试开始的时间
    start_time = datetime.datetime.now()
    print(f"Run {unique_prefix} is scheduled as {run_arn}:")
    
    # 初始化测试运行状态和结果变量
    state = "UNKNOWN"
    result = ""
    try:
        # 循环检查测试运行状态，直到测试完成
        while True:
            r = client.get_run(arn=run_arn)
            state = r["run"]["status"]

            if state == "COMPLETED":
                result = r["run"]["result"]
                break

            # 计算测试等待时间，并打印当前状态信息
            waiting_time = datetime.datetime.now() - start_time
            print(
                f"Run {unique_prefix} in state {state} after {datetime.datetime.now() - start_time}"
            )
            time.sleep(30)
    except Exception as error:
        # 如果发生异常，记录警告信息并退出程序
        warnings.warn(f"Failed to run {unique_prefix}: {error}")
        sys.exit(1)

    # 如果测试结果为空或者为 "FAILED"，打印失败信息并退出程序
    if not result or result == "FAILED":
        print(f"Run {unique_prefix} failed, exiting...")
        sys.exit(1)
# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    # 调用主函数 main()
    main()
```