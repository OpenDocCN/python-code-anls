# `MetaGPT\metagpt\tools\ut_writer.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 导入所需的模块
import json
from pathlib import Path

# 导入自定义的模块
from metagpt.provider.openai_api import OpenAILLM as GPTAPI
from metagpt.utils.common import awrite

# 定义一个接口示例
ICL_SAMPLE = """Interface definition:

Interface Name: Element Tagging
Interface Path: /projects/{project_key}/node-tags
Method: POST

Request parameters:
Path parameters:
project_key

Body parameters:
Name	Type	Required	Default Value	Remarks
nodes	array	Yes		Nodes
	node_key	string	No		Node key
	tags	array	No		Original node tag list
	node_type	string	No		Node type DATASET / RECIPE
operations	array	Yes		
	tags	array	No		Operation tag list
	mode	string	No		Operation type ADD / DELETE

Return data:
Name	Type	Required	Default Value	Remarks
code	integer	Yes		Status code
msg	string	Yes		Prompt message
data	object	Yes		Returned data
list	array	No		Node list true / false
node_type	string	No		Node type DATASET / RECIPE
node_key	string	No		Node key


Unit test：

# 定义一个参数化测试函数，用于测试接口
@pytest.mark.parametrize(
"project_key, nodes, operations, expected_msg",
[
("project_key", [{"node_key": "dataset_001", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "success"),
("project_key", [{"node_key": "dataset_002", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["tag1"], "mode": "DELETE"}], "success"),
("", [{"node_key": "dataset_001", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "Missing the required parameter project_key"),
(123, [{"node_key": "dataset_001", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "Incorrect parameter type"),
("project_key", [{"node_key": "a"*201, "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "Request parameter exceeds field boundary")
]
)
# 定义一个空的测试函数，用于接收参数化测试
def test_node_tags(project_key, nodes, operations, expected_msg):
    pass

# 上面是一个接口定义和一个单元测试示例。
# 接下来，请扮演一位在谷歌有20年经验的专业测试经理的角色。当我给出接口定义时，回复我一个单元测试。有几个要求：
# 1. 只输出一个 `@pytest.mark.parametrize` 和相应的 test_<interface name> 函数（在 pass 里面，不要实现）。
# -- 函数参数包含用于结果验证的 expected_msg。
# 2. 生成的测试用例使用更短的文本或数字，并尽可能紧凑。
# 3. 如果需要注释，请使用中文。

# 如果你理解了，请等我给出接口定义，只回答“Understood”以节省令牌。
"""

# 定义一些提示信息
ACT_PROMPT_PREFIX = """Refer to the test types: such as missing request parameters, field boundary verification, incorrect field type.
Please output 10 test cases within one `@pytest.mark.parametrize` scope.

"""

YFT_PROMPT_PREFIX = """Refer to the test types: such as SQL injection, cross-site scripting (XSS), unauthorized access and privilege escalation, 
authentication and authorization, parameter verification, exception handling, file upload and download.
Please output 10 test cases within one `@pytest.mark.parametrize` scope.

"""

# 定义一个接口示例
OCR_API_DOC = """```text
Interface Name: OCR recognition
Interface Path: /api/v1/contract/treaty/task/ocr
Method: POST

Request Parameters:
Path Parameters:

Body Parameters:
Name	Type	Required	Default Value	Remarks
file_id	string	Yes		
box	array	Yes		
contract_id	number	Yes		Contract id
start_time	string	No		yyyy-mm-dd
end_time	string	No		yyyy-mm-dd
extract_type	number	No		Recognition type 1- During import 2- After import Default 1

Response Data:
Name	Type	Required	Default Value	Remarks
code	integer	Yes		
message	string	Yes		
data	object	Yes		

"""

```