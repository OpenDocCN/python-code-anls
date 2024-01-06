# `kubehunter\kube_hunter\modules\report\dispatchers.py`

```
# 导入 logging、os 和 requests 模块
import logging
import os
import requests

# 获取当前模块的 logger 对象
logger = logging.getLogger(__name__)

# 定义一个 HTTPDispatcher 类
class HTTPDispatcher(object):
    # 定义 dispatch 方法，用于发送报告
    def dispatch(self, report):
        # 记录调试信息
        logger.debug("Dispatching report via HTTP")
        # 获取环境变量 KUBEHUNTER_HTTP_DISPATCH_METHOD 的值，如果不存在则默认为 "POST"
        dispatch_method = os.environ.get("KUBEHUNTER_HTTP_DISPATCH_METHOD", "POST").upper()
        # 获取环境变量 KUBEHUNTER_HTTP_DISPATCH_URL 的值，如果不存在则默认为 "https://localhost/"
        dispatch_url = os.environ.get("KUBEHUNTER_HTTP_DISPATCH_URL", "https://localhost/")
        # 尝试发送 HTTP 请求
        try:
            # 发起 HTTP 请求，使用 dispatch_method 指定的方法，发送 report 数据到 dispatch_url
            r = requests.request(
                dispatch_method, dispatch_url, json=report, headers={"Content-Type": "application/json"},
            )
            # 如果请求返回的状态码不是 2xx，则抛出异常
            r.raise_for_status()
            # 记录信息日志，报告已发送到 dispatch_url
            logger.info(f"Report was dispatched to: {dispatch_url}")
            # 记录调试信息，显示请求的状态码和返回的内容
            logger.debug(f"Dispatch responded {r.status_code} with: {r.text}")
        # 捕获异常
        except Exception as e:
            # 记录错误日志
            logger.error(f"Failed to dispatch report: {e}")
        except requests.HTTPError:
            # 如果发生 HTTP 错误，记录异常信息
            logger.exception(f"Failed making HTTP {dispatch_method} to {dispatch_url}, " f"status code {r.status_code}")
        except Exception:
            # 如果发生其他异常，记录异常信息
            logger.exception(f"Could not dispatch report to {dispatch_url}")


class STDOUTDispatcher(object):
    # 定义一个 STDOUTDispatcher 类
    def dispatch(self, report):
        # 将报告通过标准输出进行分发
        logger.debug("Dispatching report via stdout")
        print(report)
```