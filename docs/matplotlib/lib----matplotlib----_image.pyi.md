# `D:\src\scipysrc\matplotlib\lib\matplotlib\_image.pyi`

```py
# 导入requests库，用于发送HTTP请求和处理响应
import requests

# 定义函数fetch_weather，接收参数city
def fetch_weather(city):
    # 使用API请求天气数据，返回响应对象
    response = requests.get(f'http://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q={city}')
    # 将JSON格式的响应体转换为Python字典
    data = response.json()
    # 提取并返回天气数据中的温度信息
    return data['current']['temp_c']

# 调用函数fetch_weather，传入城市名London并打印结果
print(fetch_weather('London'))
```