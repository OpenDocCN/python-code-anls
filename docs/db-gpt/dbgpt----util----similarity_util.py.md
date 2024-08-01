# `.\DB-GPT-src\dbgpt\util\similarity_util.py`

```py
"""Utility functions for calculating similarity."""
# 导入必要的类型检查和数据结构
from typing import TYPE_CHECKING, Any, List, Sequence

# 如果在类型检查模式下，导入Embeddings类型
if TYPE_CHECKING:
    from dbgpt.core.interface.embeddings import Embeddings


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate the cosine similarity between two vectors.

    Args:
        embedding1(List[float]): The first vector.
        embedding2(List[float]): The second vector.

    Returns:
        float: The cosine similarity.
    """
    # 尝试导入numpy库，如果失败则抛出ImportError
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required for SimilarityMetric")
    
    # 计算两个向量的点积
    dot_product = np.dot(embedding1, embedding2)
    # 计算向量的范数（模长）
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    return similarity


def sigmoid_function(x: float) -> float:
    """Calculate the sigmoid function.

    The sigmoid function is defined as:
    .. math::
        f(x) = \\frac{1}{1 + e^{-x}}

    It is used to map the input to a value between 0 and 1.

    Args:
        x(float): The input to the sigmoid function.

    Returns:
        float: The output of the sigmoid function.
    """
    # 尝试导入numpy库，如果失败则抛出ImportError
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required for sigmoid_function")
    
    # 计算sigmoid函数的输出
    return 1 / (1 + np.exp(-x))


def calculate_cosine_similarity(
    embeddings: "Embeddings", prediction: str, contexts: Sequence[str]
) -> Any:
    """Calculate the cosine similarity between a prediction and a list of contexts.

    Args:
        embeddings(Embeddings): The embeddings to use.
        prediction(str): The prediction.
        contexts(Sequence[str]): The contexts.

    Returns:
        numpy.ndarray: The cosine similarity.
    """
    # 尝试导入numpy库，如果失败则抛出ImportError
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required for SimilarityMetric")
    
    # 将预测转换为嵌入向量
    prediction_vec = np.asarray(embeddings.embed_query(prediction)).reshape(1, -1)
    # 将上下文列表转换为嵌入向量列表
    context_list = list(contexts)
    context_list_vec = np.asarray(embeddings.embed_documents(context_list)).reshape(
        len(contexts), -1
    )
    # 计算预测向量和上下文向量列表的余弦相似度
    dot = np.dot(context_list_vec, prediction_vec.T).reshape(
        -1,
    )
    # 计算上下文向量列表和预测向量的范数乘积
    norm = np.linalg.norm(context_list_vec, axis=1) * np.linalg.norm(
        prediction_vec, axis=1
    )
    # 返回余弦相似度
    return dot / norm
```