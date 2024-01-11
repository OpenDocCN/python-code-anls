# `kubehunter\kube_hunter\modules\report\dispatchers.py`

```
# 导入日志、操作系统和请求模块
import logging
import os
import requests

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# 定义 HTTPDispatcher 类
class HTTPDispatcher(object):
    # 定义 dispatch 方法，用于发送报告
    def dispatch(self, report):
        # 记录调试信息
        logger.debug("Dispatching report via HTTP")
        # 获取环境变量中的 HTTP 发送方法，默认为 POST
        dispatch_method = os.environ.get("KUBEHUNTER_HTTP_DISPATCH_METHOD", "POST").upper()
        # 获取环境变量中的 HTTP 发送地址，默认为 https://localhost/
        dispatch_url = os.environ.get("KUBEHUNTER_HTTP_DISPATCH_URL", "https://localhost/")
        try:
            # 发送 HTTP 请求
            r = requests.request(
                dispatch_method, dispatch_url, json=report, headers={"Content-Type": "application/json"},
            )
            # 如果请求返回错误状态码，则抛出异常
            r.raise_for_status()
            # 记录信息日志，报告已发送至哪个地址
            logger.info(f"Report was dispatched to: {dispatch_url}")
            # 记录调试信息，显示响应状态码和内容
            logger.debug(f"Dispatch responded {r.status_code} with: {r.text}")

        # 如果请求返回 HTTP 错误，则记录异常信息
        except requests.HTTPError:
            logger.exception(f"Failed making HTTP {dispatch_method} to {dispatch_url}, " f"status code {r.status_code}")
        # 如果发生其他异常，则记录异常信息
        except Exception:
            logger.exception(f"Could not dispatch report to {dispatch_url}")


# 定义 STDOUTDispatcher 类
class STDOUTDispatcher(object):
    # 定义 dispatch 方法，用于在标准输出中打印报告
    def dispatch(self, report):
        # 记录调试信息
        logger.debug("Dispatching report via stdout")
        # 在标准输出中打印报告
        print(report)
```