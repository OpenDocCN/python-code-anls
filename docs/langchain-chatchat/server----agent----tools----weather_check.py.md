# `.\Langchain-Chatchat\server\agent\tools\weather_check.py`

```
"""
更简单的单参数输入工具实现，用于查询现在天气的情况
"""

# 导入必要的模块
from pydantic import BaseModel, Field
import requests
from configs.kb_config import SENIVERSE_API_KEY

# 定义查询天气的函数，传入地点和 API 密钥作为参数
def weather(location: str, api_key: str):
    # 构建查询天气的 API URL
    url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={location}&language=zh-Hans&unit=c"
    # 发送 GET 请求获取天气数据
    response = requests.get(url)
    # 如果响应状态码为 200，表示请求成功
    if response.status_code == 200:
        # 解析 JSON 格式的响应数据
        data = response.json()
        # 提取温度和天气描述信息
        weather = {
            "temperature": data["results"][0]["now"]["temperature"],
            "description": data["results"][0]["now"]["text"],
        }
        # 返回天气信息
        return weather
    else:
        # 如果请求失败，抛出异常
        raise Exception(
            f"Failed to retrieve weather: {response.status_code}")

# 定义一个简化查询天气的函数，只需要传入地点参数
def weathercheck(location: str):
    return weather(location, SENIVERSE_API_KEY)

# 定义天气输入的数据模型
class WeatherInput(BaseModel):
    location: str = Field(description="City name,include city and county")
```