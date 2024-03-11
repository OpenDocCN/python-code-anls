# `.\Langchain-Chatchat\server\model_workers\SparkApi.py`

```
import base64
import datetime
import hashlib
import hmac
from urllib.parse import urlparse
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

# 定义一个类用于处理请求参数
class Ws_Param(object):
    # 初始化方法，接收四个参数：APPID, APIKey, APISecret, Spark_url
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        # 从Spark_url中解析出host和path
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    # 生成url的方法
    def create_url(self):
        # 获取当前时间
        now = datetime.now()
        # 将时间转换为RFC1123格式
        date = format_date_time(mktime(now.timetuple()))

        # 拼接签名原文
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 使用hmac-sha256算法对签名原文进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        # 对加密后的结果进行base64编码
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        # 构建authorization_origin字符串
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        # 对authorization_origin进行base64编码
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成最终的url
        url = self.Spark_url + '?' + urlencode(v)
        # 返回生成的url
        return url

# 定义一个函数用于生成请求参数
def gen_params(appid, domain, question, temperature, max_token):
    """
    通过appid和用户的提问来生成请求参数
    """
    # 定义一个包含不同部分的数据字典
    data = {
        # 定义头部信息
        "header": {
            "app_id": appid,  # 应用程序ID
            "uid": "1234"  # 用户ID
        },
        # 定义参数信息
        "parameter": {
            "chat": {
                "domain": domain,  # 领域
                "random_threshold": 0.5,  # 随机阈值
                "max_tokens": max_token,  # 最大标记数
                "auditing": "default",  # 审计
                "temperature": temperature,  # 温度
            }
        },
        # 定义负载信息
        "payload": {
            "message": {
                "text": question  # 文本信息
            }
        }
    }
    # 返回整个数据字典
    return data
```