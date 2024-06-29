# `D:\src\scipysrc\pandas\pandas\tests\indexes\ranges\__init__.py`

```
# 导入 requests 库，用于发送 HTTP 请求
import requests

# 定义一个名为 get_weather 的函数，用于获取指定城市的天气信息
def get_weather(city):
    # 构建请求的 URL，包含城市名和 API key
    url = f'http://api.weatherstack.com/current?access_key=YOUR_API_KEY&query={city}'
    
    # 发送 GET 请求到天气 API，并获取响应对象
    response = requests.get(url)
    
    # 如果响应状态码不是 200（即成功），打印错误信息并返回空字典
    if response.status_code != 200:
        print(f'Error {response.status_code} fetching weather for {city}')
        return {}
    
    # 从响应中获取 JSON 格式的天气数据
    weather_data = response.json()
    
    # 如果返回的数据中有 'current' 字段，则返回该字段对应的内容；否则返回空字典
    return weather_data.get('current', {})
```