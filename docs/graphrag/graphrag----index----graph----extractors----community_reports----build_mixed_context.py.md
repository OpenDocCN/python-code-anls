# `.\graphrag\graphrag\index\graph\extractors\community_reports\build_mixed_context.py`

```py
# 版权声明和许可信息，版权归 Microsoft Corporation 所有，基于 MIT 许可证发布
"""定义了 build_mixed_context 方法的模块。"""

# 导入 pandas 库，用于数据处理
import pandas as pd

# 导入自定义模块，包括图形提取和社区报告的模式定义
import graphrag.index.graph.extractors.community_reports.schemas as schemas

# 导入自定义模块，用于处理文本工具函数
from graphrag.query.llm.text_utils import num_tokens

# 导入本地模块，用于排序上下文
from .sort_context import sort_context


def build_mixed_context(context: list[dict], max_tokens: int) -> str:
    """
    构建混合上下文，通过连接所有子社区的上下文。

    如果上下文超出限制，我们使用子社区报告代替。
    """
    # 根据上下文大小排序，从大到小排列
    sorted_context = sorted(
        context, key=lambda x: x[schemas.CONTEXT_SIZE], reverse=True
    )

    # 用子社区报告替换本地上下文，从最大的子社区开始
    substitute_reports = []  # 替代报告列表
    final_local_contexts = []  # 最终的本地上下文列表
    exceeded_limit = True  # 是否超过了上限
    context_string = ""  # 上下文字符串初始化为空

    # 遍历排序后的所有子社区上下文
    for idx, sub_community_context in enumerate(sorted_context):
        if exceeded_limit:
            # 如果子社区上下文有完整内容，则加入替代报告列表
            if sub_community_context[schemas.FULL_CONTENT]:
                substitute_reports.append({
                    schemas.COMMUNITY_ID: sub_community_context[schemas.SUB_COMMUNITY],
                    schemas.FULL_CONTENT: sub_community_context[schemas.FULL_CONTENT],
                })
            else:
                # 如果子社区没有报告，则使用其本地上下文
                final_local_contexts.extend(sub_community_context[schemas.ALL_CONTEXT])
                continue

            # 添加剩余子社区的本地上下文
            remaining_local_context = []
            for rid in range(idx + 1, len(sorted_context)):
                remaining_local_context.extend(sorted_context[rid][schemas.ALL_CONTEXT])
            
            # 构建新的上下文字符串，通过排序上下文函数生成
            new_context_string = sort_context(
                local_context=remaining_local_context + final_local_contexts,
                sub_community_reports=substitute_reports,
            )
            # 检查新上下文字符串的长度是否符合最大标记数限制
            if num_tokens(new_context_string) <= max_tokens:
                exceeded_limit = False
                context_string = new_context_string
                break

    # 如果所有子社区报告都超出了限制，则添加报告直到上下文完全填满
    if exceeded_limit:
        substitute_reports = []
        for sub_community_context in sorted_context:
            substitute_reports.append({
                schemas.COMMUNITY_ID: sub_community_context[schemas.SUB_COMMUNITY],
                schemas.FULL_CONTENT: sub_community_context[schemas.FULL_CONTENT],
            })
            # 将替代报告转换为 CSV 格式的字符串
            new_context_string = pd.DataFrame(substitute_reports).to_csv(
                index=False, sep=","
            )
            # 检查新上下文字符串的长度是否超出最大标记数限制
            if num_tokens(new_context_string) > max_tokens:
                break

            context_string = new_context_string

    # 返回构建好的上下文字符串
    return context_string
```