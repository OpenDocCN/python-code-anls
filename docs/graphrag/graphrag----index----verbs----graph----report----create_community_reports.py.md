# `.\graphrag\graphrag\index\verbs\graph\report\create_community_reports.py`

```py
    # 引入日志记录模块
    log = logging.getLogger(__name__)

    # 定义一个枚举类，表示创建社区报告的策略类型
    class CreateCommunityReportsStrategyType(str, Enum):
        """CreateCommunityReportsStrategyType class definition."""

        # 策略类型为图智能
        graph_intelligence = "graph_intelligence"

        # 返回策略类型的字符串表示形式
        def __repr__(self):
            """Get a string representation."""
            return f'"{self.value}"'

    # 使用装饰器定义一个异步函数，用于创建社区报告
    @verb(name="create_community_reports")
    async def create_community_reports(
        input: VerbInput,
        callbacks: VerbCallbacks,
        cache: PipelineCache,
        strategy: dict,
        async_mode: AsyncType = AsyncType.AsyncIO,
        num_threads: int = 4,
        **_kwargs,
    ) -> TableContainer:
        """Generate entities for each row, and optionally a graph of those entities."""
        
        # 记录调试信息，输出当前策略的内容
        log.debug("create_community_reports strategy=%s", strategy)
        
        # 获取输入的本地上下文数据，转换为 DataFrame 类型
        local_contexts = cast(pd.DataFrame, input.get_input())
        
        # 获取名为 "nodes" 的必需输入表格，并转换为 DataFrame 类型
        nodes_ctr = get_required_input_table(input, "nodes")
        nodes = cast(pd.DataFrame, nodes_ctr.table)
        
        # 获取名为 "community_hierarchy" 的必需输入表格，并转换为 DataFrame 类型
        community_hierarchy_ctr = get_required_input_table(input, "community_hierarchy")
        community_hierarchy = cast(pd.DataFrame, community_hierarchy_ctr.table)
        
        # 获取节点层级信息
        levels = get_levels(nodes)
        
        # 初始化报告列表，每个元素是 CommunityReport 或 None 类型
        reports: list[CommunityReport | None] = []
        
        # 设置进度回调函数，用于显示处理进度
        tick = progress_ticker(callbacks.progress, len(local_contexts))
        
        # 加载指定策略类型的运行器
        runner = load_strategy(strategy["type"])
    # 对于每个层级(level)遍历执行以下操作
    for level in levels:
        # 准备社区报告的上下文信息
        level_contexts = prep_community_report_context(
            pd.DataFrame(reports),  # 使用报告数据创建 Pandas 数据框
            local_context_df=local_contexts,  # 本地上下文数据框
            community_hierarchy_df=community_hierarchy,  # 社区层级结构数据框
            level=level,  # 当前处理的层级
            max_tokens=strategy.get(  # 最大输入标记数，使用策略中的值或默认值
                "max_input_tokens", defaults.COMMUNITY_REPORT_MAX_INPUT_LENGTH
            ),
        )

        # 定义一个异步函数，用于生成报告
        async def run_generate(record):
            # 调用 _generate_report 异步函数生成报告结果
            result = await _generate_report(
                runner,
                community_id=record[schemas.NODE_COMMUNITY],  # 社区 ID
                community_level=record[schemas.COMMUNITY_LEVEL],  # 社区层级
                community_context=record[schemas.CONTEXT_STRING],  # 社区上下文字符串
                cache=cache,  # 缓存对象
                callbacks=callbacks,  # 回调函数
                strategy=strategy,  # 策略参数
            )
            tick()  # 记录时间点
            return result  # 返回生成的报告结果

        # 根据给定的行数据(level_contexts)，并发执行 run_generate 函数
        local_reports = await derive_from_rows(
            level_contexts,  # 上下文数据
            run_generate,  # 生成报告的异步函数
            callbacks=NoopVerbCallbacks(),  # 空操作的回调函数
            num_threads=num_threads,  # 线程数量
            scheduling_type=async_mode,  # 异步模式类型
        )
        # 将生成的本地报告扩展到总报告列表中
        reports.extend([lr for lr in local_reports if lr is not None])

    # 返回包含报告数据的数据框的表格容器
    return TableContainer(table=pd.DataFrame(reports))
# 异步函数，生成单个社区的报告
async def _generate_report(
    runner: CommunityReportsStrategy,            # 运行报告生成的策略对象
    cache: PipelineCache,                       # 管道缓存对象，用于存储中间结果
    callbacks: VerbCallbacks,                   # 回调函数集合，用于报告生成过程中的通知和反馈
    strategy: dict,                             # 报告生成策略的配置字典
    community_id: int | str,                    # 社区的唯一标识符，可以是整数或字符串
    community_level: int,                       # 社区的级别，通常是一个整数
    community_context: str,                     # 社区的上下文信息，描述社区的环境或特征
) -> CommunityReport | None:                    # 返回一个社区报告对象或者空值

    """Generate a report for a single community."""  # 生成单个社区的报告的函数说明文档
    return await runner(
        community_id, community_context, community_level, callbacks, cache, strategy
    )


def load_strategy(
    strategy: CreateCommunityReportsStrategyType,  # 策略类型枚举值，指定要加载的报告生成策略
) -> CommunityReportsStrategy:                    # 返回一个社区报告生成策略对象

    """Load strategy method definition."""         # 加载报告生成策略的函数说明文档
    match strategy:                               # 匹配传入的策略类型枚举值
        case CreateCommunityReportsStrategyType.graph_intelligence:  # 如果是图智能策略
            from .strategies.graph_intelligence import run  # 导入图智能策略的运行函数
            return run                             # 返回该策略的运行函数
        case _:                                   # 如果是其他未知的策略类型
            msg = f"Unknown strategy: {strategy}"  # 构造未知策略的错误消息
            raise ValueError(msg)                  # 抛出值错误异常，表示策略未知或不支持
```