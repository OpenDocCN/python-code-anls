# `.\transformers\benchmark\__init__.py`

```
import requests  # 导入 requests 模块，用于发送 HTTP 请求

# 定义函数 download_image，用于下载图片
def download_image(url):
    # 发送 HTTP GET 请求获取图片数据
    response = requests.get(url)
    # 将图片数据保存到本地文件
    with open('image.jpg', 'wb') as f:
        f.write(response.content)

# 调用 download_image 函数，传入图片 URL
download_image('https://example.com/image.jpg')
```