# `.\AutoGPT\benchmark\agbenchmark\challenges\verticals\code\4_url_shortener\artifacts_out\url_shortener.py`

```py
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 base64 模块，用于 URL 编码
import base64

# 创建 URL 映射字典
URL_MAPPING = {}

# 缩短 URL 的函数
def shorten_url(url):
    # 将 URL 转换为 base64 编码
    encoded_url = base64.b64encode(url.encode()).decode()
    # 取编码后的 URL 的前 8 个字符作为缩短后的 URL
    short_url = encoded_url[:8]
    # 将缩短后的 URL 映射回原始 URL
    URL_MAPPING[short_url] = url
    return short_url

# 检索 URL 的函数
def retrieve_url(short_url):
    return URL_MAPPING.get(short_url, "URL not found")

# 主函数
def main():
    # 创建 ArgumentParser 对象，设置描述信息
    parser = argparse.ArgumentParser(description="URL Shortener")
    # 添加命令行参数选项，用于缩短 URL
    parser.add_argument("-s", "--shorten", type=str, help="URL to be shortened")
    # 添加命令行参数选项，用于检索缩短后的 URL
    parser.add_argument("-r", "--retrieve", type=str, help="Short URL to be retrieved")

    # 解析命令行参数
    args = parser.parse_args()

    # 如果存在缩短 URL 的参数
    if args.shorten:
        # 缩短 URL 并打印结果
        shortened_url = shorten_url(args.shorten)
        print(shortened_url)
        # 直接使用新缩短的 URL 进行检索
        print(retrieve_url(shortened_url))
    # 如果存在检索 URL 的参数
    elif args.retrieve:
        # 打印检索结果
        print(retrieve_url(args.retrieve))
    # 如果没有提供有效参数
    else:
        print("No valid arguments provided.")

# 当作为脚本直接运行时执行主函数
if __name__ == "__main__":
    main()
```