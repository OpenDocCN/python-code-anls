# `.\translation\src\populate.py`

```
# 导入日志模块
import logging

# 导入 pandas 库，并重命名为 pd
import pandas as pd

# 导入自定义模块中的 Document 类
from langchain.docstore.document import Document

# 从自定义模块中导入 Encoder 类
from encoder.encoder import Encoder

# 从自定义模块中导入 VectorDatabase 类
from retriever.vector_db import VectorDatabase

# 设置日志配置，设定日志级别为 INFO
logging.basicConfig(level=logging.INFO)

# 获取或创建名为 logger 的日志器
logger = logging.getLogger()

# 如果脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":

    # 输出日志信息，指示正在初始化 Encoder 和 VectorDatabase 类
    logger.info("Init encoder and vectorDB classes..")

    # 创建 Encoder 实例
    encoder = Encoder()

    # 使用 Encoder 实例化 VectorDatabase，创建 vectordb 对象
    vectordb = VectorDatabase(encoder.encoder)

    # 输出日志信息，指示正在加载数据
    logger.info("Loading data..")

    # 使用 pandas 读取名为 data.csv 的文件内容，并存储在 DataFrame df 中
    df = pd.read_csv("data/data.csv")

    # 输出日志信息，指示正在创建完整评论列
    logger.info("Creating full review column..")

    # 对 DataFrame 中的 "reviews.title" 和 "reviews.text" 列执行 lambda 函数，将结果连接成一个字符串，存入 "full_review" 列
    df["full_review"] = df[["reviews.title", "reviews.text"]].apply(
        lambda row: ". ".join(row.values.astype(str)), axis=1
    )

    # 输出日志信息，指示正在将评论存储在 vectorDB 中
    logger.info("Storing reviews in vectorDB..")

    # 遍历 df["asins"] 列中的唯一值的前 10 个产品 ID
    for product_id in df["asins"].unique()[:10]:
        # 创建 Document 对象列表，每个对象包含对应 product_id 的所有完整评论
        docs = [
            Document(page_content=item)
            for item in df[df["asins"] == product_id]["full_review"].tolist()
        ]

        # 从 Document 对象列表中创建 passages
        passages = vectordb.create_passages_from_documents(docs)

        # 将 passages 存储到数据库中，使用 product_id 作为键
        vectordb.store_passages_db(passages, product_id)
```