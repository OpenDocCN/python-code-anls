# `.\pytorch\test\distributed\tensor\parallel\__init__.py`

```
# 导入requests库，用于发送HTTP请求
import requests

# 定义函数download_image，接收参数url和filename，用于下载图片到本地
def download_image(url, filename):
    # 发送GET请求获取图片数据
    response = requests.get(url)
    # 将获取到的图片数据写入本地文件
    with open(filename, 'wb') as f:
        f.write(response.content)
```