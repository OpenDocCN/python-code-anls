# `.\pytorch\tools\__init__.py`

```
# 导入 requests 库，用于发送 HTTP 请求
import requests

# 定义函数 get_weather，接收参数 city
def get_weather(city):
    # 构造 API 请求 URL，包含城市名
    url = f'http://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q={city}&aqi=no'
    
    # 发送 GET 请求到 API，并获取响应对象
    response = requests.get(url)
    
    # 如果响应状态码为 200（成功）
    if response.status_code == 200:
        # 从响应中获取 JSON 数据并转换为 Python 字典
        data = response.json()
        
        # 从返回的数据字典中提取当前天气的描述信息
        weather_desc = data['current']['condition']['text']
        
        # 返回天气描述信息
        return weather_desc
    
    # 如果请求失败，输出错误信息并返回空字符串
    else:
        print(f"Error fetching weather data for {city}. Status code: {response.status_code}")
        return ''
```