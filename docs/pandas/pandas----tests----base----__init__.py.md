# `D:\src\scipysrc\pandas\pandas\tests\base\__init__.py`

```
# 导入所需的模块：json（处理JSON格式数据）、urllib.request（处理URL请求）
import json
import urllib.request

# 定义一个函数 fetch_weather，接收城市名称作为参数
def fetch_weather(city):
    # 拼接URL，使用OpenWeatherMap的API查询城市天气
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid=your_api_key'
    # 使用urllib.request.urlopen打开URL，并读取响应内容
    response = urllib.request.urlopen(url)
    # 读取响应内容，并解码为UTF-8格式的字符串
    data = response.read().decode('utf-8')
    # 解析JSON格式的字符串数据，转换为Python字典对象
    weather = json.loads(data)
    # 返回包含天气信息的字典对象
    return weather
```