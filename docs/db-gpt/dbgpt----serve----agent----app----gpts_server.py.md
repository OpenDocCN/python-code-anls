# `.\DB-GPT-src\dbgpt\serve\agent\app\gpts_server.py`

```py
# 从 dbgpt._private.config 模块导入 Config 类
# 从 dbgpt.component 模块导入 ComponentType 类
# 从 dbgpt.model.cluster 模块导入 BaseModelController 类
# 从 dbgpt.serve.agent.db.gpts_app 模块导入 GptsAppCollectionDao 和 GptsAppDao 类
from dbgpt._private.config import Config
from dbgpt.component import ComponentType
from dbgpt.model.cluster import BaseModelController
from dbgpt.serve.agent.db.gpts_app import GptsAppCollectionDao, GptsAppDao

# 创建 Config 类的实例，保存到 CFG 变量中
CFG = Config()

# 创建 GptsAppCollectionDao 类的实例，保存到 collection_dao 变量中
collection_dao = GptsAppCollectionDao()
# 创建 GptsAppDao 类的实例，保存到 gpts_dao 变量中
gpts_dao = GptsAppDao()


# 异步函数定义：获取可用的 LLMS（语言模型服务）工作者名称列表
async def available_llms(worker_type: str = "llm"):
    # 从 CFG 实例的 SYSTEM_APP 组件中获取 MODEL_CONTROLLER 组件的实例，类型为 BaseModelController
    controller = CFG.SYSTEM_APP.get_component(
        ComponentType.MODEL_CONTROLLER, BaseModelController
    )
    # 创建一个空集合 types，用于存储不重复的工作者名称
    types = set()
    # 获取所有健康状态的模型实例列表
    models = await controller.get_all_instances(healthy_only=True)
    # 遍历模型实例列表
    for model in models:
        # 将模型名称按 "@" 符号拆分成 worker_name 和 wt 两部分
        worker_name, wt = model.model_name.split("@")
        # 如果模型的工作者类型 wt 等于传入的 worker_type
        if wt == worker_type:
            # 将 worker_name 添加到 types 集合中
            types.add(worker_name)
    # 将 types 集合转换为列表并返回
    return list(types)
```