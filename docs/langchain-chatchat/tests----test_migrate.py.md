# `.\Langchain-Chatchat\tests\test_migrate.py`

```py
# 导入必要的模块
from pathlib import Path
from pprint import pprint
import os
import shutil
import sys
# 获取当前文件的父目录的父目录作为根路径
root_path = Path(__file__).parent.parent
# 将根路径添加到系统路径中
sys.path.append(str(root_path))

# 导入知识库相关的模块和函数
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.utils import get_kb_path, get_doc_path, KnowledgeFile
from server.knowledge_base.migrate import folder2db, prune_db_docs, prune_folder_files

# 设置测试知识库的名称和测试文件
kb_name = "test_kb_for_migrate"
test_files = {
    "readme.md": str(root_path / "readme.md"),
}

# 获取知识库路径和文档路径
kb_path = get_kb_path(kb_name)
doc_path = get_doc_path(kb_name)

# 如果文档路径不存在，则创建
if not os.path.isdir(doc_path):
    os.makedirs(doc_path)

# 将测试文件复制到文档路径中
for k, v in test_files.items():
    shutil.copy(v, os.path.join(doc_path, k))

# 测试重新创建版本库
def test_recreate_vs():
    # 将文件夹中的文件导入到数据库中，重新创建版本库
    folder2db([kb_name], "recreate_vs")

    # 获取知识库服务对象
    kb = KBServiceFactory.get_service_by_name(kb_name)
    # 断言知识库存在
    assert kb and kb.exists()

    # 列出知识库中的文件
    files = kb.list_files()
    print(files)
    for name in test_files:
        assert name in files
        path = os.path.join(doc_path, name)

        # 根据文件名列出文档
        docs = kb.list_docs(file_name=name)
        assert len(docs) > 0
        pprint(docs[0])
        for doc in docs:
            assert doc.metadata["source"] == name

        # 根据元数据列出文档
        docs = kb.list_docs(metadata={"source": name})
        assert len(docs) > 0

        for doc in docs:
            assert doc.metadata["source"] == name

# 测试增量导入
def test_increment():
    # 获取知识库服务对象
    kb = KBServiceFactory.get_service_by_name(kb_name)
    # 清空版本库
    kb.clear_vs()
    assert kb.list_files() == []
    assert kb.list_docs() == []

    # 将文件夹中的文件增量导入到数据库中
    folder2db([kb_name], "increment")

    # 列出知识库中的文件
    files = kb.list_files()
    print(files)
    for f in test_files:
        assert f in files

        # 根据文件名列出文档
        docs = kb.list_docs(file_name=f)
        assert len(docs) > 0
        pprint(docs[0])

        for doc in docs:
            assert doc.metadata["source"] == f

# 测试修剪数据库
def test_prune_db():
    # 获取要删除和保留的文件名
    del_file, keep_file = list(test_files)[:2
    # 删除指定路径下的文件
    os.remove(os.path.join(doc_path, del_file))
    
    # 删除数据库中指定知识库的文档
    prune_db_docs([kb_name])
    
    # 根据知识库名称获取知识库服务对象
    kb = KBServiceFactory.get_service_by_name(kb_name)
    
    # 获取知识库中的所有文件列表
    files = kb.list_files()
    print(files)
    
    # 确保被删除的文件不在文件列表中
    assert del_file not in files
    # 确保被保留的文件在文件列表中
    assert keep_file in files
    
    # 获取知识库中指定文件名的文档列表
    docs = kb.list_docs(file_name=del_file)
    # 确保被删除的文件没有对应的文档
    assert len(docs) == 0
    
    # 获取知识库中指定文件名的文档列表
    docs = kb.list_docs(file_name=keep_file)
    # 确保被保留的文件有对应的文档
    assert len(docs) > 0
    # 打印第一个文档的内容
    pprint(docs[0])
    
    # 复制测试文件到指定路径下
    shutil.copy(test_files[del_file], os.path.join(doc_path, del_file))
# 测试删除文件夹中的文件
def test_prune_folder():
    # 从测试文件列表中选择要删除的文件和要保留的文件
    del_file, keep_file = list(test_files)[:2]
    # 根据知识库名称获取知识库服务
    kb = KBServiceFactory.get_service_by_name(kb_name)

    # 删除指定文件的文档
    kb.delete_doc(KnowledgeFile(del_file, kb_name))
    # 获取知识库中的文件列表
    files = kb.list_files()
    print(files)
    # 断言被删除的文件不在文件列表中
    assert del_file not in files
    # 断言要保留的文件在文件列表中
    assert keep_file in files

    # 获取指定文件名的文档列表
    docs = kb.list_docs(file_name=del_file)
    # 断言被删除的文件没有对应的文档
    assert len(docs) == 0

    # 获取指定文件名的文档列表
    docs = kb.list_docs(file_name=keep_file)
    # 断言要保留的文件有对应的文档
    assert len(docs) > 0

    # 再次获取被删除文件名的文档列表
    docs = kb.list_docs(file_name=del_file)
    # 断言被删除的文件没有对应的文档
    assert len(docs) == 0

    # 断言指定路径下存在被删除的文件
    assert os.path.isfile(os.path.join(doc_path, del_file))

    # 清理文件夹
    prune_folder_files([kb_name])

    # 检查结果
    # 断言被删除的文件不再存在于指定路径下
    assert not os.path.isfile(os.path.join(doc_path, del_file))
    # 断言要保留的文件存在于指定路径下
    assert os.path.isfile(os.path.join(doc_path, keep_file))


# 测试删除知识库
def test_drop_kb():
    # 根据知识库名称获取知识库服务
    kb = KBServiceFactory.get_service_by_name(kb_name)
    # 删除知识库
    kb.drop_kb()
    # 断言知识库不存在
    assert not kb.exists()
    # 断言知识库路径不存在
    assert not os.path.isdir(kb_path)

    # 再次根据知识库名称获取知识库服务
    kb = KBServiceFactory.get_service_by_name(kb_name)
    # 断言知识库服务为None
    assert kb is None
```