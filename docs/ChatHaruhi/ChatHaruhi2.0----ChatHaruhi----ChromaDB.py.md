# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\ChromaDB.py`

```py
import chromadb
from .BaseDB import BaseDB
import random
import string
import os

class ChromaDB(BaseDB):
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.path = None
    
    def init_db(self):
        # 如果数据库客户端已经初始化，则打印信息并返回
        if self.client is not None:
            print('ChromaDB has already been initialized')
            return

        folder_name = ''

        # 创建一个未存在的名为temp_<随机字符串>的文件夹
        while os.path.exists(folder_name) or folder_name == '':
            folder_name =  "tempdb_" + ''.join(random.sample(string.ascii_letters + string.digits, 8))

        self.path = folder_name
        self.client = chromadb.PersistentClient(path=folder_name)

        # 获取或创建名为"search"的集合
        self.collection = self.client.get_or_create_collection("search")

    def save(self, file_path):
        # 如果file_path与当前路径不同
        if file_path != self.path:
            # 将self.path下的所有文件复制到file_path，覆盖已存在的文件
            os.system("cp -r " + self.path + " " + file_path)
            previous_path = self.path
            self.path = file_path
            self.client = chromadb.PersistentClient(path=file_path)
            # 如果之前的路径以"tempdb"开头，则删除该路径下的所有文件
            if previous_path.startswith("tempdb"):
                os.system("rm -rf " + previous_path)

    def load(self, file_path):
        # 加载指定路径的数据库
        self.path = file_path
        self.client = chromadb.PersistentClient(path=file_path)
        # 获取名为"search"的集合
        self.collection = self.client.get_collection("search")

    def search(self, vector, n_results):
        # 使用给定的向量进行搜索，并返回结果中的第一个文档
        results = self.collection.query(query_embeddings=[vector], n_results=n_results)
        return results['documents'][0]

    def init_from_docs(self, vectors, documents):
        # 如果客户端未初始化，则初始化数据库
        if self.client is None:
            self.init_db()
        
        ids = []
        # 对每个文档生成唯一标识符，并将其添加到ids列表中
        for i, doc in enumerate(documents):
            first_four_chat = doc[:min(4, len(doc))]
            ids.append(str(i) + "_" + doc)
        # 向集合中添加文档及其向量
        self.collection.add(embeddings=vectors, documents=documents, ids=ids)
```