# `.\safetensors_conversion.py`

```
import json  # 导入json模块，用于处理JSON格式数据
import uuid  # 导入uuid模块，用于生成唯一标识符
from typing import Optional  # 导入Optional类型，用于可选的类型声明

import requests  # 导入requests模块，用于发送HTTP请求
from huggingface_hub import Discussion, HfApi, get_repo_discussions  # 导入huggingface_hub相关函数和类

from .utils import cached_file, logging  # 从当前包中导入cached_file和logging模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象


def previous_pr(api: HfApi, model_id: str, pr_title: str, token: str) -> Optional["Discussion"]:
    # 获取主提交的commit_id
    main_commit = api.list_repo_commits(model_id, token=token)[0].commit_id
    # 遍历当前模型repo中的所有讨论
    for discussion in get_repo_discussions(repo_id=model_id, token=token):
        # 判断讨论是否为打开的PR并且标题为pr_title
        if discussion.title == pr_title and discussion.status == "open" and discussion.is_pull_request:
            # 获取与讨论相关的提交信息
            commits = api.list_repo_commits(model_id, revision=discussion.git_reference, token=token)

            # 检查主提交是否与PR的第二个提交相同
            if main_commit == commits[1].commit_id:
                return discussion  # 如果条件符合，返回此讨论对象
    return None  # 如果未找到符合条件的讨论，返回None


def spawn_conversion(token: str, private: bool, model_id: str):
    logger.info("Attempting to convert .bin model on the fly to safetensors.")

    safetensors_convert_space_url = "https://safetensors-convert.hf.space"
    sse_url = f"{safetensors_convert_space_url}/queue/join"
    sse_data_url = f"{safetensors_convert_space_url}/queue/data"

    # 指定fn_index以指示使用Space的run方法
    hash_data = {"fn_index": 1, "session_hash": str(uuid.uuid4())}

    def start(_sse_connection, payload):
        # 迭代SSE连接的每一行数据
        for line in _sse_connection.iter_lines():
            line = line.decode()
            if line.startswith("data:"):
                resp = json.loads(line[5:])  # 解析收到的JSON数据
                logger.debug(f"Safetensors conversion status: {resp['msg']}")
                # 处理不同的转换状态
                if resp["msg"] == "queue_full":
                    raise ValueError("Queue is full! Please try again.")
                elif resp["msg"] == "send_data":
                    event_id = resp["event_id"]
                    # 发送数据到sse_data_url
                    response = requests.post(
                        sse_data_url,
                        stream=True,
                        params=hash_data,
                        json={"event_id": event_id, **payload, **hash_data},
                    )
                    response.raise_for_status()  # 检查响应状态
                elif resp["msg"] == "process_completed":
                    return  # 如果转换完成，结束函数

    with requests.get(sse_url, stream=True, params=hash_data) as sse_connection:
        data = {"data": [model_id, private, token]}
        try:
            logger.debug("Spawning safetensors automatic conversion.")
            start(sse_connection, data)  # 调用start函数开始转换
        except Exception as e:
            logger.warning(f"Error during conversion: {repr(e)}")  # 处理转换过程中的异常情况


def get_conversion_pr_reference(api: HfApi, model_id: str, **kwargs):
    private = api.model_info(model_id).private  # 获取模型信息中的private字段值

    logger.info("Attempting to create safetensors variant")
    pr_title = "Adding `safetensors` variant of this model"
    token = kwargs.get("token")

    # 这段代码查找当前repo中是否有关于safetensors的已打开的PR
    # 调用函数 `previous_pr`，获取先前创建的 pull request 对象
    pr = previous_pr(api, model_id, pr_title, token=token)

    # 如果 pr 为 None 或者（不是私有且 pr 的作者不是 "SFConvertBot"），则执行以下操作：
    if pr is None or (not private and pr.author != "SFConvertBot"):
        # 调用函数 `spawn_conversion`，启动转换过程
        spawn_conversion(token, private, model_id)
        # 再次获取先前创建的 pull request 对象
        pr = previous_pr(api, model_id, pr_title, token=token)
    else:
        # 记录日志，指示安全张量的 pull request 已存在
        logger.info("Safetensors PR exists")

    # 构建 SHA 引用，格式为 "refs/pr/{pr.num}"
    sha = f"refs/pr/{pr.num}"

    # 返回 SHA 引用
    return sha
# 自动转换函数，根据预训练模型名称或路径以及其他缓存文件参数来执行自动转换
def auto_conversion(pretrained_model_name_or_path: str, **cached_file_kwargs):
    # 使用给定的 token 创建 Hugging Face API 的实例
    api = HfApi(token=cached_file_kwargs.get("token"))
    
    # 获取转换 Pull Request 的参考 SHA 值
    sha = get_conversion_pr_reference(api, pretrained_model_name_or_path, **cached_file_kwargs)

    # 如果没有找到 SHA 值，则返回 None
    if sha is None:
        return None, None
    
    # 将 SHA 值添加到缓存文件参数中的 revision 键中
    cached_file_kwargs["revision"] = sha
    
    # 从缓存文件参数中删除 _commit_hash 键
    del cached_file_kwargs["_commit_hash"]

    # 这是一个额外的 HEAD 调用，如果能从 PR 描述中推断出分片/非分片，可以删除这个调用
    # 检查指定的模型是否存在分片的 "model.safetensors.index.json" 文件
    sharded = api.file_exists(
        pretrained_model_name_or_path,
        "model.safetensors.index.json",
        revision=sha,
        token=cached_file_kwargs.get("token"),
    )
    
    # 根据是否存在分片文件，选择相应的文件名
    filename = "model.safetensors.index.json" if sharded else "model.safetensors"

    # 缓存解析后的归档文件
    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
    
    # 返回解析后的归档文件路径、SHA 值和是否分片的标志
    return resolved_archive_file, sha, sharded
```