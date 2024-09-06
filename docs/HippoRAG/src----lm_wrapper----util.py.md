# `.\HippoRAG\src\lm_wrapper\util.py`

```py
# 初始化嵌入模型
def init_embedding_model(model_name):
    # 如果模型名称中包含 'GritLM/'，则导入 GritWrapper 并返回其实例
    if 'GritLM/' in model_name:
        from src.lm_wrapper.gritlm import GritWrapper
        return GritWrapper(model_name)
    # 如果模型名称不在指定的 ['colbertv2', 'bm25'] 中，则导入 HuggingFaceWrapper 并返回其实例
    elif model_name not in ['colbertv2', 'bm25']:
        from src.lm_wrapper.huggingface_util import HuggingFaceWrapper
        return HuggingFaceWrapper(model_name)  # 返回 HuggingFace 模型用于检索
```