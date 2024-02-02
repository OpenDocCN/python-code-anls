# `kubehunter\kube_hunter\modules\report\factory.py`

```py
# 从 kube_hunter.modules.report.json 模块中导入 JSONReporter 类
from kube_hunter.modules.report.json import JSONReporter
# 从 kube_hunter.modules.report.yaml 模块中导入 YAMLReporter 类
from kube_hunter.modules.report.yaml import YAMLReporter
# 从 kube_hunter.modules.report.plain 模块中导入 PlainReporter 类
from kube_hunter.modules.report.plain import PlainReporter
# 从 kube_hunter.modules.report.dispatchers 模块中导入 STDOUTDispatcher, HTTPDispatcher 类
from kube_hunter.modules.report.dispatchers import STDOUTDispatcher, HTTPDispatcher

# 导入 logging 模块
import logging

# 获取当前模块的 logger 对象
logger = logging.getLogger(__name__)

# 设置默认的报告格式为 "plain"
DEFAULT_REPORTER = "plain"
# 创建报告格式到对应类的映射字典
reporters = {
    "yaml": YAMLReporter,
    "json": JSONReporter,
    "plain": PlainReporter,
}

# 设置默认的分发器为 "stdout"
DEFAULT_DISPATCHER = "stdout"
# 创建分发器到对应类的映射字典
dispatchers = {
    "stdout": STDOUTDispatcher,
    "http": HTTPDispatcher,
}

# 根据报告格式名称获取对应的报告类实例
def get_reporter(name):
    try:
        return reporters[name.lower()]()
    except KeyError:
        # 如果报告格式名称不存在，则记录警告并使用默认的报告格式
        logger.warning(f'Unknown reporter "{name}", using f{DEFAULT_REPORTER}')
        return reporters[DEFAULT_REPORTER]()

# 根据分发器名称获取对应的分发器类实例
def get_dispatcher(name):
    try:
        return dispatchers[name.lower()]()
    except KeyError:
        # 如果分发器名称不存在，则记录警告并使用默认的分发器
        logger.warning(f'Unknown dispatcher "{name}", using {DEFAULT_DISPATCHER}')
        return dispatchers[DEFAULT_DISPATCHER]()
```