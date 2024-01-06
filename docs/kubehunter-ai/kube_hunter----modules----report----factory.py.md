# `kubehunter\kube_hunter\modules\report\factory.py`

```
# 导入所需的模块和类
from kube_hunter.modules.report.json import JSONReporter
from kube_hunter.modules.report.yaml import YAMLReporter
from kube_hunter.modules.report.plain import PlainReporter
from kube_hunter.modules.report.dispatchers import STDOUTDispatcher, HTTPDispatcher

# 导入日志模块
import logging

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 设置默认的报告格式为 plain
DEFAULT_REPORTER = "plain"
# 创建报告格式和对应的报告类的映射关系
reporters = {
    "yaml": YAMLReporter,
    "json": JSONReporter,
    "plain": PlainReporter,
}

# 设置默认的报告分发方式为 stdout
DEFAULT_DISPATCHER = "stdout"
# 创建报告分发方式和对应的分发类的映射关系
dispatchers = {
    "stdout": STDOUTDispatcher,
    "http": HTTPDispatcher,
}
# 获取指定名称的报告生成器，如果不存在则返回默认报告生成器
def get_reporter(name):
    try:
        # 尝试从报告生成器字典中获取对应名称的生成器并返回
        return reporters[name.lower()]()
    except KeyError:
        # 如果名称不存在，则记录警告并返回默认报告生成器
        logger.warning(f'Unknown reporter "{name}", using f{DEFAULT_REPORTER}')
        return reporters[DEFAULT_REPORTER]()

# 获取指定名称的调度器，如果不存在则返回默认调度器
def get_dispatcher(name):
    try:
        # 尝试从调度器字典中获取对应名称的调度器并返回
        return dispatchers[name.lower()]()
    except KeyError:
        # 如果名称不存在，则记录警告并返回默认调度器
        logger.warning(f'Unknown dispatcher "{name}", using {DEFAULT_DISPATCHER}')
        return dispatchers[DEFAULT_DISPATCHER]()
```