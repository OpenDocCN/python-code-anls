# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\lightning\__init__.py`

```
# 导入requests库，用于发送HTTP请求
import requests

# 定义一个名为get_weather的函数，接受一个参数city
def get_weather(city):
    # 构建API请求的URL，使用city参数进行查询
    url = f'http://api.weatherstack.com/current?access_key=your_api_key&query={city}'
    
    # 发送GET请求到构建的URL，并获取响应对象
    response = requests.get(url)
    
    # 从响应对象中获取JSON格式的数据
    data = response.json()
    
    # 从JSON数据中提取当前天气信息的字典
    current_weather = data['current']
    
    # 提取当前天气信息中的温度
    temperature = current_weather['temperature']
    
    # 返回当前城市的温度
    return temperature
```