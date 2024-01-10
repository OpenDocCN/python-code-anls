# `so-vits-svc\export_index_for_onnx.py`

```
# 导入操作系统和 pickle 模块
import os
import pickle

# 导入 faiss 模块
import faiss

# 设置路径和文件名
path = "crs"
indexs_file_path = f"checkpoints/{path}/feature_and_index.pkl"
indexs_out_dir = f"checkpoints/{path}/"

# 以二进制读取 feature_and_index.pkl 文件
with open("feature_and_index.pkl",mode="rb") as f:
    # 从文件中加载索引数据
    indexs = pickle.load(f)

# 遍历索引数据
for k in indexs:
    # 打印保存索引的信息
    print(f"Save {k} index")
    # 将索引写入文件
    faiss.write_index(
        indexs[k],
        os.path.join(indexs_out_dir,f"Index-{k}.index")
    )

# 打印保存完所有索引的信息
print("Saved all index")
```