# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\BaseDB.py`

```py
# BaseDB.py

# 导入抽象基类（ABC）和抽象方法（abstractmethod）装饰器
from abc import ABC, abstractmethod

# 定义名为 BaseDB 的抽象基类（ABC）
class BaseDB(ABC):

    # 抽象方法：初始化数据库，无具体实现
    @abstractmethod
    def init_db(self):
        pass
    
    # 抽象方法：保存数据到指定文件路径，无具体实现
    @abstractmethod
    def save(self, file_path):
        pass

    # 抽象方法：从指定文件路径加载数据，无具体实现
    @abstractmethod
    def load(self, file_path):
        pass

    # 抽象方法：根据向量进行搜索，返回指定数量的结果，无具体实现
    @abstractmethod
    def search(self, vector, n_results):
        pass

    # 抽象方法：从文档初始化数据库，无具体实现
    @abstractmethod
    def init_from_docs(self, vectors, documents):
        pass
```