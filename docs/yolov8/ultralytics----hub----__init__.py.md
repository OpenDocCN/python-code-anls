# `.\yolov8\ultralytics\hub\__init__.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

import requests  # 导入requests库，用于发送HTTP请求

from ultralytics.data.utils import HUBDatasetStats  # 导入HUBDatasetStats工具类
from ultralytics.hub.auth import Auth  # 导入Auth类，用于认证
from ultralytics.hub.session import HUBTrainingSession  # 导入HUBTrainingSession类，用于处理训练会话
from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX, events  # 导入常量和事件
from ultralytics.utils import LOGGER, SETTINGS, checks  # 导入日志记录器、设置和检查工具

__all__ = (
    "PREFIX",
    "HUB_WEB_ROOT",
    "HUBTrainingSession",
    "login",
    "logout",
    "reset_model",
    "export_fmts_hub",
    "export_model",
    "get_export",
    "check_dataset",
    "events",
)


def login(api_key: str = None, save=True) -> bool:
    """
    Log in to the Ultralytics HUB API using the provided API key.

    The session is not stored; a new session is created when needed using the saved SETTINGS or the HUB_API_KEY
    environment variable if successfully authenticated.

    Args:
        api_key (str, optional): API key to use for authentication.
            If not provided, it will be retrieved from SETTINGS or HUB_API_KEY environment variable.
        save (bool, optional): Whether to save the API key to SETTINGS if authentication is successful.

    Returns:
        (bool): True if authentication is successful, False otherwise.
    """
    checks.check_requirements("hub-sdk>=0.0.8")  # 检查是否满足SDK的最低版本要求
    from hub_sdk import HUBClient  # 导入HUBClient类来进行HUB API的客户端操作

    api_key_url = f"{HUB_WEB_ROOT}/settings?tab=api+keys"  # 设置API密钥设置页面的重定向URL
    saved_key = SETTINGS.get("api_key")  # 获取保存在SETTINGS中的API密钥
    active_key = api_key or saved_key  # 使用提供的API密钥或从环境变量中获取的API密钥

    credentials = {"api_key": active_key} if active_key and active_key != "" else None  # 设置认证凭据

    client = HUBClient(credentials)  # 初始化HUBClient客户端对象

    if client.authenticated:
        # 成功通过HUB认证

        if save and client.api_key != saved_key:
            SETTINGS.update({"api_key": client.api_key})  # 更新SETTINGS中的有效API密钥

        # 根据是否提供了API密钥或从设置中检索到来设置消息内容
        log_message = (
            "New authentication successful ✅" if client.api_key == api_key or not credentials else "Authenticated ✅"
        )
        LOGGER.info(f"{PREFIX}{log_message}")  # 记录认证成功信息到日志

        return True
    else:
        # 未能通过HUB认证
        LOGGER.info(f"{PREFIX}Get API key from {api_key_url} and then run 'yolo hub login API_KEY'")
        return False


def logout():
    """
    Log out of Ultralytics HUB by removing the API key from the settings file. To log in again, use 'yolo hub login'.

    Example:
        ```py
        from ultralytics import hub

        hub.logout()
        ```py
    """
    SETTINGS["api_key"] = ""  # 清空SETTINGS中的API密钥
    SETTINGS.save()  # 保存SETTINGS变更
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo hub login'.")  # 记录退出登录信息到日志


def reset_model(model_id=""):
    """Reset a trained model to an untrained state."""
    r = requests.post(f"{HUB_API_ROOT}/model-reset", json={"modelId": model_id}, headers={"x-api-key": Auth().api_key})
    # 发送POST请求到HUB API以重置指定model_id的模型为未训练状态
    # 检查 HTTP 响应状态码是否为 200
    if r.status_code == 200:
        # 如果响应状态码为 200，记录信息日志，表示模型重置成功
        LOGGER.info(f"{PREFIX}Model reset successfully")
        # 返回空，结束函数执行
        return
    
    # 如果响应状态码不为 200，记录警告日志，表示模型重置失败，并包含响应的状态码和原因
    LOGGER.warning(f"{PREFIX}Model reset failure {r.status_code} {r.reason}")
def export_fmts_hub():
    """Returns a list of HUB-supported export formats."""
    # 导入 export_formats 函数，该函数位于 ultralytics.engine.exporter 模块中
    from ultralytics.engine.exporter import export_formats
    # 返回 export_formats 函数返回值的第二个元素至最后一个元素（不包括第一个元素），并添加两个特定的输出格式
    return list(export_formats()["Argument"][1:]) + ["ultralytics_tflite", "ultralytics_coreml"]


def export_model(model_id="", format="torchscript"):
    """Export a model to all formats."""
    # 断言指定的导出格式在支持的格式列表中，如果不支持则抛出 AssertionError
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    # 发起 POST 请求，导出指定模型到指定格式，并使用 API 密钥进行身份验证
    r = requests.post(
        f"{HUB_API_ROOT}/v1/models/{model_id}/export", json={"format": format}, headers={"x-api-key": Auth().api_key}
    )
    # 断言请求的状态码为 200，否则抛出 AssertionError，显示错误信息
    assert r.status_code == 200, f"{PREFIX}{format} export failure {r.status_code} {r.reason}"
    # 记录导出操作开始的信息
    LOGGER.info(f"{PREFIX}{format} export started ✅")


def get_export(model_id="", format="torchscript"):
    """Get an exported model dictionary with download URL."""
    # 断言指定的导出格式在支持的格式列表中，如果不支持则抛出 AssertionError
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    # 发起 POST 请求，获取导出的模型字典及其下载链接，并使用 API 密钥进行身份验证
    r = requests.post(
        f"{HUB_API_ROOT}/get-export",
        json={"apiKey": Auth().api_key, "modelId": model_id, "format": format},
        headers={"x-api-key": Auth().api_key},
    )
    # 断言请求的状态码为 200，否则抛出 AssertionError，显示错误信息
    assert r.status_code == 200, f"{PREFIX}{format} get_export failure {r.status_code} {r.reason}"
    # 返回从响应中解析得到的 JSON 格式的导出模型字典
    return r.json()


def check_dataset(path: str, task: str) -> None:
    """
    Function for error-checking HUB dataset Zip file before upload. It checks a dataset for errors before it is uploaded
    to the HUB. Usage examples are given below.

    Args:
        path (str): Path to data.zip (with data.yaml inside data.zip).
        task (str): Dataset task. Options are 'detect', 'segment', 'pose', 'classify', 'obb'.

    Example:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
            i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.
        ```py
        from ultralytics.hub import check_dataset

        check_dataset('path/to/coco8.zip', task='detect')  # detect dataset
        check_dataset('path/to/coco8-seg.zip', task='segment')  # segment dataset
        check_dataset('path/to/coco8-pose.zip', task='pose')  # pose dataset
        check_dataset('path/to/dota8.zip', task='obb')  # OBB dataset
        check_dataset('path/to/imagenet10.zip', task='classify')  # classification dataset
        ```py
    """
    # 使用 HUBDatasetStats 类检查指定路径下的数据集文件（zip 格式），并为指定任务类型生成 JSON 格式的统计信息
    HUBDatasetStats(path=path, task=task).get_json()
    # 记录检查操作成功完成的信息
    LOGGER.info(f"Checks completed correctly ✅. Upload this dataset to {HUB_WEB_ROOT}/datasets/.")
```