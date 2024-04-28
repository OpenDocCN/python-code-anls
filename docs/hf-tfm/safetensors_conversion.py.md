# `.\transformers\safetensors_conversion.py`

```
# 导入所需模块
import json  # 导入 JSON 模块，用于处理 JSON 数据
import uuid  # 导入 uuid 模块，用于生成唯一标识符
from typing import Optional  # 从 typing 模块导入 Optional 类型，用于函数参数和返回类型注解

import requests  # 导入 requests 模块，用于发送 HTTP 请求
from huggingface_hub import Discussion, HfApi, get_repo_discussions  # 从 huggingface_hub 模块导入相关函数

# 导入自定义的工具函数和日志记录器
from .utils import cached_file, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


# 函数：查找先前的 Pull Request
def previous_pr(api: HfApi, model_id: str, pr_title: str, token: str) -> Optional["Discussion"]:
    # 获取主要提交的 commit_id
    main_commit = api.list_repo_commits(model_id, token=token)[0].commit_id
    # 遍历仓库的讨论列表
    for discussion in get_repo_discussions(repo_id=model_id, token=token):
        # 如果讨论是 Pull Request 且标题与给定的标题相同且状态为打开
        if discussion.title == pr_title and discussion.status == "open" and discussion.is_pull_request:
            # 获取该 PR 的提交列表
            commits = api.list_repo_commits(model_id, revision=discussion.git_reference, token=token)
            # 如果主要提交与该 PR 的第二个提交的 commit_id 相同
            if main_commit == commits[1].commit_id:
                return discussion  # 返回找到的 PR
    return None  # 如果未找到符合条件的 PR，则返回 None


# 函数：触发转换过程
def spawn_conversion(token: str, private: bool, model_id: str):
    # 记录日志信息
    logger.info("Attempting to convert .bin model on the fly to safetensors.")

    # 定义安全张量转换服务的 URL
    safetensors_convert_space_url = "https://safetensors-convert.hf.space"
    # 定义服务器端推送事件的 URL
    sse_url = f"{safetensors_convert_space_url}/queue/join"
    # 定义数据推送的 URL
    sse_data_url = f"{safetensors_convert_space_url}/queue/data"

    # 构建数据哈希信息，包含函数索引和会话哈希
    hash_data = {"fn_index": 1, "session_hash": str(uuid.uuid4())}

    # 定义启动转换过程的函数
    def start(_sse_connection, payload):
        # 从服务器端推送事件连接迭代返回的行
        for line in _sse_connection.iter_lines():
            line = line.decode()  # 将字节流解码为字符串
            if line.startswith(""):  # 如果行以""开头
                # 解析 JSON 数据
                resp = json.loads(line[5:])
                # 记录转换状态
                logger.debug(f"Safetensors conversion status: {resp['msg']}")
                # 如果队列已满，则抛出异常
                if resp["msg"] == "queue_full":
                    raise ValueError("Queue is full! Please try again.")
                # 如果需要发送数据，则发送数据
                elif resp["msg"] == "send_data":
                    event_id = resp["event_id"]
                    # 发送 POST 请求，将数据发送到服务器端
                    response = requests.post(
                        sse_data_url,
                        stream=True,
                        params=hash_data,
                        json={"event_id": event_id, **payload, **hash_data},
                    )
                    response.raise_for_status()  # 如果请求不成功，抛出异常
                # 如果处理完成，则返回
                elif resp["msg"] == "process_completed":
                    return

    # 使用 SSE 连接访问转换服务
    with requests.get(sse_url, stream=True, params=hash_data) as sse_connection:
        # 构建要发送的数据
        data = {"data": [model_id, private, token]}
        try:
            logger.debug("Spawning safetensors automatic conversion.")  # 记录调试信息
            start(sse_connection, data)  # 启动转换过程
        except Exception as e:
            logger.warning(f"Error during conversion: {repr(e)}")  # 记录转换过程中的异常信息


# 函数：获取转换 PR 的引用
def get_conversion_pr_reference(api: HfApi, model_id: str, **kwargs):
    # 获取模型是否为私有模型
    private = api.model_info(model_id).private

    # 记录日志信息
    logger.info("Attempting to create safetensors variant")
    # 定义 PR 的标题
    pr_title = "Adding `safetensors` variant of this model"
    # 获取访问令牌
    token = kwargs.get("token")

    # 这部分代码用于查找当前仓库中是否存在针对 safetensors 的 PR
    # 如果存在，它会检查该 PR 是否已经打开
    # 如果已经打开，则会检查该 PR 的提交是否是最新的主要提交
    # 调用函数previous_pr()获取之前的Pull Request对象，用于检查是否存在之前的PR
    pr = previous_pr(api, model_id, pr_title, token=token)

    # 如果之前的PR不存在或者（不是私有且PR作者不是"SFConvertBot"），则执行以下操作
    if pr is None or (not private and pr.author != "SFConvertBot"):
        # 调用spawn_conversion()函数生成转换任务
        spawn_conversion(token, private, model_id)
        # 重新获取之前的PR对象
        pr = previous_pr(api, model_id, pr_title, token=token)
    else:
        # 如果之前的PR存在，则记录日志信息
        logger.info("Safetensors PR exists")

    # 生成SHA引用，用于标识PR
    sha = f"refs/pr/{pr.num}"

    # 返回SHA引用
    return sha
# 自动转换预训练模型名称或路径
def auto_conversion(pretrained_model_name_or_path: str, **cached_file_kwargs):
    # 创建 Hugging Face API 对象
    api = HfApi(token=cached_file_kwargs.get("token"))
    # 获取转换 PR 的参考 SHA
    sha = get_conversion_pr_reference(api, pretrained_model_name_or_path, **cached_file_kwargs)

    # 如果没有找到 SHA，则返回 None
    if sha is None:
        return None, None
    # 将 SHA 添加到缓存文件参数中
    cached_file_kwargs["revision"] = sha
    # 删除缓存文件参数中的 _commit_hash
    del cached_file_kwargs["_commit_hash"]

    # 这是一个额外的 HEAD 调用，如果能够从 PR 描述中推断出分片/非分片，则可以删除
    # 检查模型是否分片
    sharded = api.file_exists(
        pretrained_model_name_or_path,
        "model.safetensors.index.json",
        revision=sha,
        token=cached_file_kwargs.get("token"),
    )
    # 根据是否分片选择文件名
    filename = "model.safetensors.index.json" if sharded else "model.safetensors"

    # 缓存预训练模型文件
    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
    # 返回缓存的文件、SHA 和是否分片的信息
    return resolved_archive_file, sha, sharded
```