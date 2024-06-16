# `.\ollama_knowlage_graph\helpers\prompts.py`

```
# 导入系统模块
import sys
# 导入 yachalk 模块中的 chalk 对象
from yachalk import chalk
# 将上级目录添加到模块搜索路径中
sys.path.append("..")

# 导入 JSON 模块
import json
# 导入 ollama 客户端模块中的 client 对象
import ollama.client as client

# 定义函数，用于提取给定上下文中的关键概念
def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    # 定义系统提示信息，包括提取关键概念的任务说明和期望的输出格式
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    # 使用 ollama 客户端生成文本
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        # 尝试解析响应的 JSON 数据
        result = json.loads(response)
        # 将元数据添加到每个结果字典中
        result = [dict(item, **metadata) for item in result]
    except:
        # 如果解析失败，打印错误信息，并将结果置为 None
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    # 返回结果
    return result

# 定义函数，用于生成图形提示
def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    # 如果未提供模型，则使用默认模型
    if model == None:
        model = "mistral-openorca:latest"

    # 获取模型信息并打印
    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))
    # 设置系统提示信息，指导用户在提取给定上下文中的术语及其关系时的操作步骤和输出格式
    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
        "\tTerms may include object, entity, location, organization, person, \n"
        "\tcondition, acronym, documents, service, concept, etc.\n"
        "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
        "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
        "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )
    
    # 设置用户提示信息，用于格式化用户输入和系统输出的模板
    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    
    # 通过API客户端生成模型输出，提供系统提示、用户输入和模型名称
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    
    try:
        # 尝试解析模型生成的JSON响应
        result = json.loads(response)
        # 将每个输出元素与额外的元数据结合为单个字典，组成最终结果列表
        result = [dict(item, **metadata) for item in result]
    except:
        # 捕获可能的异常情况，并输出错误信息和响应内容以辅助调试
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    
    # 返回处理后的结果列表
    return result
```