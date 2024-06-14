# `.\demo\flaskdemo.py`

```
import json  # 导入处理 JSON 数据的模块
import argparse  # 导入命令行参数解析模块
import torch  # 导入 PyTorch 深度学习框架
import os  # 导入操作系统相关功能的模块
import random  # 导入生成随机数的模块
import numpy as np  # 导入数值计算库 NumPy
import requests  # 导入处理 HTTP 请求的模块
import logging  # 导入日志记录模块
import math  # 导入数学运算函数的模块
import copy  # 导入复制对象的模块
import string  # 导入处理字符串的模块

from tqdm import tqdm  # 从 tqdm 模块导入进度条显示组件
from time import time  # 从 time 模块导入时间函数 time
from flask import Flask, request, jsonify  # 导入 Flask 框架及相关组件
from flask_cors import CORS  # 导入处理跨域资源共享的模块
from tornado.wsgi import WSGIContainer  # 导入 Tornado 的 WSGI 容器
from tornado.httpserver import HTTPServer  # 导入 Tornado 的 HTTP 服务器
from tornado.ioloop import IOLoop  # 导入 Tornado 的 I/O 循环

from simcse import SimCSE  # 从自定义模块 simcse 导入 SimCSE 类

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)  # 配置日志格式和级别
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

def run_simcse_demo(port, args):
    app = Flask(__name__, static_folder='./static')  # 创建 Flask 应用实例，设置静态文件夹路径
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # 设置 Flask 应用返回 JSON 格式数据时不使用美化输出
    CORS(app)  # 处理跨域资源共享

    sentence_path = os.path.join(args.sentences_dir, args.example_sentences)  # 拼接示例句子文件路径
    query_path = os.path.join(args.sentences_dir, args.example_query)  # 拼接查询示例文件路径
    embedder = SimCSE(args.model_name_or_path)  # 根据模型路径创建 SimCSE 对象
    embedder.build_index(sentence_path)  # 构建索引，用于示例查询

    @app.route('/')
    def index():
        return app.send_static_file('index.html')  # 返回静态文件 index.html

    @app.route('/api', methods=['GET'])
    def api():
        query = request.args['query']  # 获取请求中的查询参数
        top_k = int(request.args['topk'])  # 获取请求中的返回结果数目参数
        threshold = float(request.args['threshold'])  # 获取请求中的相似度阈值参数
        start = time()  # 记录处理开始时间
        results = embedder.search(query, top_k=top_k, threshold=threshold)  # 使用 SimCSE 进行查询
        ret = []  # 初始化返回结果列表
        out = {}  # 初始化输出字典
        for sentence, score in results:
            ret.append({"sentence": sentence, "score": score})  # 将查询结果格式化为指定格式并加入结果列表
        span = time() - start  # 计算处理时间
        out['ret'] = ret  # 将结果列表加入输出字典
        out['time'] = "{:.4f}".format(span)  # 将处理时间加入输出字典，并保留四位小数
        return jsonify(out)  # 返回 JSON 格式的输出字典

    @app.route('/files/<path:path>')
    def static_files(path):
        return app.send_static_file('files/' + path)  # 返回静态文件夹中指定路径的文件

    @app.route('/get_examples', methods=['GET'])
    def get_examples():
        with open(query_path, 'r') as fp:
            examples = [line.strip() for line in fp.readlines()]  # 读取示例查询文件中的内容并存入列表
        return jsonify(examples)  # 返回 JSON 格式的示例查询列表

    addr = args.ip + ":" + args.port  # 构建服务器地址
    logger.info(f'Starting Index server at {addr}')  # 记录服务器启动信息
    http_server = HTTPServer(WSGIContainer(app))  # 创建基于 WSGI 的 HTTP 服务器
    http_server.listen(port)  # 监听指定端口
    IOLoop.instance().start()  # 启动 Tornado 的 I/O 循环

if __name__=="__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--model_name_or_path', default=None, type=str)  # 添加模型路径参数
    parser.add_argument('--device', default='cpu', type=str)  # 添加设备参数，默认为 CPU
    parser.add_argument('--sentences_dir', default=None, type=str)  # 添加示例句子目录路径参数
    parser.add_argument('--example_query', default=None, type=str)  # 添加示例查询文件名参数
    parser.add_argument('--example_sentences', default=None, type=str)  # 添加示例句子文件名参数
    parser.add_argument('--port', default='8888', type=str)  # 添加服务器端口参数，默认为 8888
    parser.add_argument('--ip', default='http://127.0.0.1')  # 添加服务器 IP 地址参数，默认为本地地址
    parser.add_argument('--load_light', default=False, action='store_true')  # 添加轻量加载标志参数
    args = parser.parse_args()  # 解析命令行参数

    run_simcse_demo(args.port, args)  # 调用函数运行 SimCSE 示例演示服务器
```