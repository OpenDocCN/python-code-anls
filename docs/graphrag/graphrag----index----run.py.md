# `.\graphrag\graphrag\index\run.py`

```py
    # 设置日志记录器，使用当前模块名称作为日志记录器的名称
    log = logging.getLogger(__name__)

    # 异步函数，运行一个根据给定配置运行的流水线
    async def run_pipeline_with_config(
        # 接受 PipelineConfig 对象或配置文件路径作为参数
        config_or_path: PipelineConfig | str,
        # 可选参数，指定要运行的工作流列表
        workflows: list[PipelineWorkflowReference] | None = None,
        # 可选参数，作为输入的数据集 DataFrame
        dataset: pd.DataFrame | None = None,
        # 可选参数，指定要使用的存储对象
        storage: PipelineStorage | None = None,
        # 可选参数，指定要使用的缓存对象
        cache: PipelineCache | None = None,
        # 可选参数，指定工作流的回调函数对象
        callbacks: WorkflowCallbacks | None = None,
        # 可选参数，指定进度报告器对象
        progress_reporter: ProgressReporter | None = None,
        # 可选参数，指定输入后处理步骤列表
        input_post_process_steps: list[PipelineWorkflowStep] | None = None,
        # 可选参数，指定额外的动词定义
        additional_verbs: VerbDefinitions | None = None,
        # 可选参数，指定额外的工作流定义
        additional_workflows: WorkflowDefinitions | None = None,
        # 可选参数，指定要使用的表发射器列表
        emit: list[TableEmitterType] | None = None,
        # 可选参数，指定是否进行内存分析
        memory_profile: bool = False,
        # 可选参数，指定运行的标识符
        run_id: str | None = None,
        # 可选参数，指定是否为恢复运行
        is_resume_run: bool = False,
        # 其余关键字参数
        **_kwargs: dict,
    ) -> AsyncIterable[PipelineRunResult]:
        """Run a pipeline with the given config.
        异步生成器函数，根据给定的配置运行流水线并生成结果对象序列
        """
    """
    Args:
        - config_or_path - 要运行管道的配置
        - workflows - 要运行的工作流（这将覆盖配置中的设置）
        - dataset - 要在管道上运行的数据集（这将覆盖配置中的设置）
        - storage - 用于管道的存储（这将覆盖配置中的设置）
        - cache - 用于管道的缓存（这将覆盖配置中的设置）
        - reporter - 用于管道的报告生成器（这将覆盖配置中的设置）
        - input_post_process_steps - 在输入数据上运行的后处理步骤（这将覆盖配置中的设置）
        - additional_verbs - 用于管道的自定义动词
        - additional_workflows - 用于管道的自定义工作流
        - emit - 用于管道的表发射器
        - memory_profile - 是否进行内存分析的标志
        - run_id - 要开始或恢复的运行ID
    """
    if isinstance(config_or_path, str):
        log.info("Running pipeline with config %s", config_or_path)
    else:
        log.info("Running pipeline")

    # 如果未提供运行ID，则使用当前时间生成一个新的
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    # 加载管道配置文件
    config = load_pipeline_config(config_or_path)
    # 应用变量替换到配置中
    config = _apply_substitutions(config, run_id)
    # 获取根目录路径
    root_dir = config.root_dir

    # 创建存储对象
    def _create_storage(config: PipelineStorageConfigTypes | None) -> PipelineStorage:
        return load_storage(
            config
            or PipelineFileStorageConfig(base_dir=str(Path(root_dir or "") / "output"))
        )

    # 创建缓存对象
    def _create_cache(config: PipelineCacheConfigTypes | None) -> PipelineCache:
        return load_cache(config or PipelineMemoryCacheConfig(), root_dir=root_dir)

    # 创建报告生成器对象
    def _create_reporter(
        config: PipelineReportingConfigTypes | None,
    ) -> WorkflowCallbacks | None:
        return load_pipeline_reporter(config, root_dir) if config else None

    # 创建输入数据对象
    async def _create_input(
        config: PipelineInputConfigTypes | None,
    ) -> pd.DataFrame | None:
        if config is None:
            return None
        return await load_input(config, progress_reporter, root_dir)

    # 创建输入数据的后处理步骤列表
    def _create_postprocess_steps(
        config: PipelineInputConfigTypes | None,
    ) -> list[PipelineWorkflowStep] | None:
        return config.post_process if config is not None else None

    # 如果未提供进度报告器，则使用空报告器
    progress_reporter = progress_reporter or NullProgressReporter()
    # 如果未提供存储对象，则创建默认存储对象
    storage = storage or _create_storage(config.storage)
    # 如果未提供缓存对象，则创建默认缓存对象
    cache = cache or _create_cache(config.cache)
    # 如果未提供报告生成器对象，则创建默认报告生成器对象
    callbacks = callbacks or _create_reporter(config.reporting)
    # 如果未提供数据集对象，则根据配置创建输入数据集
    dataset = dataset if dataset is not None else await _create_input(config.input)
    # 如果未提供输入数据的后处理步骤，则根据配置创建后处理步骤列表
    post_process_steps = input_post_process_steps or _create_postprocess_steps(
        config.input
    )
    # 如果未提供工作流对象，则使用配置中的默认工作流
    workflows = workflows or config.workflows

    # 如果数据集为空，则抛出数值错误异常
    if dataset is None:
        msg = "No dataset provided!"
        raise ValueError(msg)
    # 使用异步迭代器运行管道，逐个处理管道返回的表格数据
    async for table in run_pipeline(
        # 指定工作流列表
        workflows=workflows,
        # 指定数据集
        dataset=dataset,
        # 指定存储方式
        storage=storage,
        # 指定缓存对象
        cache=cache,
        # 指定回调函数列表
        callbacks=callbacks,
        # 指定输入后处理步骤列表
        input_post_process_steps=post_process_steps,
        # 指定内存分析配置
        memory_profile=memory_profile,
        # 指定额外动词操作
        additional_verbs=additional_verbs,
        # 指定额外工作流列表
        additional_workflows=additional_workflows,
        # 指定进度报告器
        progress_reporter=progress_reporter,
        # 指定发射器
        emit=emit,
        # 指定是否为恢复运行
        is_resume_run=is_resume_run,
    ):
        # 逐个产生管道处理后的表格数据
        yield table
# 异步函数：运行管道流程，接受多个工作流引用、数据集、存储、缓存、回调、进度报告器等参数
async def run_pipeline(
    workflows: list[PipelineWorkflowReference],  # 工作流引用列表
    dataset: pd.DataFrame,  # 数据集，应至少包含'id'、'text'、'title'列
    storage: PipelineStorage | None = None,  # 管道使用的存储对象，默认为None
    cache: PipelineCache | None = None,  # 管道使用的缓存对象，默认为None
    callbacks: WorkflowCallbacks | None = None,  # 管道使用的回调对象，默认为None
    progress_reporter: ProgressReporter | None = None,  # 管道使用的进度报告器对象，默认为None
    input_post_process_steps: list[PipelineWorkflowStep] | None = None,  # 输入数据的后处理步骤列表，默认为None
    additional_verbs: VerbDefinitions | None = None,  # 自定义动词定义对象，默认为None
    additional_workflows: WorkflowDefinitions | None = None,  # 自定义工作流定义对象，默认为None
    emit: list[TableEmitterType] | None = None,  # 发射器类型列表，默认为None
    memory_profile: bool = False,  # 是否启用内存分析，默认为False
    is_resume_run: bool = False,  # 是否为继续运行，默认为False
    **_kwargs: dict,  # 其它未命名参数，保存在_kwargs字典中
) -> AsyncIterable[PipelineRunResult]:
    """Run the pipeline.

    Args:
        - workflows - 要运行的工作流列表
        - dataset - 要在管道上运行的数据集，特别是至少包含以下列：
            - id - 文档的ID
            - text - 文档的文本
            - title - 文档的标题
            如果有后处理步骤，则必须在此之后存在！
        - storage - 用于管道的存储对象
        - cache - 用于管道的缓存对象
        - reporter - 用于管道的报告器对象
        - input_post_process_steps - 要在输入数据上运行的后处理步骤
        - additional_verbs - 用于管道的自定义动词
        - additional_workflows - 用于管道的自定义工作流
        - debug - 是否以调试模式运行
    Returns:
        - output - 当它们完成运行时的工作流结果的可迭代对象，以及发生的任何错误
    """
    start_time = time.time()  # 记录管道开始运行的时间
    stats = PipelineRunStats()  # 创建管道运行统计对象
    storage = storage or MemoryPipelineStorage()  # 如果未提供存储对象，则使用内存存储
    cache = cache or InMemoryCache()  # 如果未提供缓存对象，则使用内存缓存
    progress_reporter = progress_reporter or NullProgressReporter()  # 如果未提供进度报告器对象，则使用空报告器
    callbacks = callbacks or ConsoleWorkflowCallbacks()  # 如果未提供回调对象，则使用控制台回调
    callbacks = _create_callback_chain(callbacks, progress_reporter)  # 创建回调链
    emit = emit or [TableEmitterType.Parquet]  # 如果未提供发射器类型列表，则默认使用Parquet格式
    # 创建表格发射器列表，处理发射错误时将调用回调对象的错误处理方法
    emitters = create_table_emitters(
        emit,
        storage,
        lambda e, s, d: cast(WorkflowCallbacks, callbacks).on_error(
            "Error emitting table", e, s, d
        ),
    )
    # 加载工作流，并可选地包括自定义动词和工作流
    loaded_workflows = load_workflows(
        workflows,
        additional_verbs=additional_verbs,
        additional_workflows=additional_workflows,
        memory_profile=memory_profile,
    )
    workflows_to_run = loaded_workflows.workflows  # 获取要运行的工作流列表
    workflow_dependencies = loaded_workflows.dependencies  # 获取工作流的依赖关系

    # 创建运行上下文，包括存储、缓存和统计信息
    context = _create_run_context(storage, cache, stats)

    if len(emitters) == 0:
        log.info(
            "No emitters provided. No table outputs will be generated. This is probably not correct."
        )

    async def dump_stats() -> None:
        await storage.set("stats.json", json.dumps(asdict(stats), indent=4))
    # 从存储加载表格数据到 pandas DataFrame 中
    async def load_table_from_storage(name: str) -> pd.DataFrame:
        # 检查存储中是否存在指定的数据集
        if not await storage.has(name):
            msg = f"Could not find {name} in storage!"
            raise ValueError(msg)
        
        try:
            # 记录日志，表明正在从存储中读取表格数据
            log.info("read table from storage: %s", name)
            # 从存储中读取 Parquet 格式的数据，并转换为 pandas DataFrame
            return pd.read_parquet(BytesIO(await storage.get(name, as_bytes=True)))
        except Exception:
            # 如果出现异常，记录异常日志并向上抛出异常
            log.exception("error loading table from storage: %s", name)
            raise

    # 向工作流注入数据依赖关系
    async def inject_workflow_data_dependencies(workflow: Workflow) -> None:
        # 向工作流中添加默认输入表格
        workflow.add_table(DEFAULT_INPUT_NAME, dataset)
        # 获取当前工作流的依赖项列表
        deps = workflow_dependencies[workflow.name]
        # 记录日志，表明正在处理当前工作流的依赖项
        log.info("dependencies for %s: %s", workflow.name, deps)
        # 遍历依赖项列表，为每个依赖项加载对应的表格数据并添加到工作流中
        for id in deps:
            workflow_id = f"workflow:{id}"
            table = await load_table_from_storage(f"{id}.parquet")
            workflow.add_table(workflow_id, table)

    # 写入工作流的统计信息
    async def write_workflow_stats(
        workflow: Workflow,
        workflow_result: WorkflowRunResult,
        workflow_start_time: float,
    ) -> None:
        # 将每个动词操作的时间记录到统计信息中
        for vt in workflow_result.verb_timings:
            stats.workflows[workflow.name][f"{vt.index}_{vt.verb}"] = vt.timing

        # 记录工作流的总运行时间
        workflow_end_time = time.time()
        stats.workflows[workflow.name]["overall"] = (
            workflow_end_time - workflow_start_time
        )
        
        # 记录整体运行时间
        stats.total_runtime = time.time() - start_time
        
        # 将统计信息持久化存储
        await dump_stats()

        # 如果有内存分析数据，保存内存分析统计信息
        if workflow_result.memory_profile is not None:
            await _save_profiler_stats(
                storage, workflow.name, workflow_result.memory_profile
            )

        # 记录调试信息，输出当前工作流的第一行数据
        log.debug(
            "first row of %s => %s", workflow_name, workflow.output().iloc[0].to_json()
        )

    # 发布工作流的输出结果
    async def emit_workflow_output(workflow: Workflow) -> pd.DataFrame:
        # 获取工作流的输出数据
        output = cast(pd.DataFrame, workflow.output())
        # 将输出数据通过所有注册的发射器进行发布
        for emitter in emitters:
            await emitter.emit(workflow.name, output)
        # 返回工作流的输出数据
        return output

    # 在完成后处理步骤后获取数据集
    dataset = await _run_post_process_steps(
        input_post_process_steps, dataset, context, callbacks
    )

    # 确保输入数据集是有效的
    _validate_dataset(dataset)

    # 记录日志，输出加载后数据集的行数
    log.info("Final # of rows loaded: %s", len(dataset))
    
    # 更新统计信息，记录数据集中文档的数量
    stats.num_documents = len(dataset)
    
    # 设置最后一个工作流的名称为 "input"
    last_workflow = "input"
    try:
        # 异步调用，记录运行统计数据
        await dump_stats()

        # 遍历待运行的工作流列表
        for workflow_to_run in workflows_to_run:
            # 尝试清理中间数据帧
            gc.collect()

            # 获取当前工作流及其名称
            workflow = workflow_to_run.workflow
            workflow_name: str = workflow.name
            last_workflow = workflow_name

            # 记录运行信息到日志
            log.info("Running workflow: %s...", workflow_name)

            # 如果是恢复运行且目标文件已存在，则跳过当前工作流
            if is_resume_run and await storage.has(
                f"{workflow_to_run.workflow.name}.parquet"
            ):
                log.info("Skipping %s because it already exists", workflow_name)
                continue

            # 初始化当前工作流的统计信息
            stats.workflows[workflow_name] = {"overall": 0.0}

            # 注入工作流数据依赖
            await inject_workflow_data_dependencies(workflow)

            # 记录工作流开始时间
            workflow_start_time = time.time()

            # 执行工作流，并获取结果
            result = await workflow.run(context, callbacks)

            # 记录工作流的统计信息
            await write_workflow_stats(workflow, result, workflow_start_time)

            # 保存工作流的输出数据
            output = await emit_workflow_output(workflow)

            # 生成工作流运行结果并返回
            yield PipelineRunResult(workflow_name, output, None)

            # 清理工作流资源
            output = None
            workflow.dispose()
            workflow = None

        # 计算总运行时间并记录统计信息
        stats.total_runtime = time.time() - start_time
        await dump_stats()

    # 捕获并记录异常信息
    except Exception as e:
        log.exception("error running workflow %s", last_workflow)

        # 调用错误处理回调函数，并生成错误信息
        cast(WorkflowCallbacks, callbacks).on_error(
            "Error running pipeline!", e, traceback.format_exc()
        )

        # 返回工作流运行结果，包含错误信息
        yield PipelineRunResult(last_workflow, None, [e])
# 创建一个回调链的管理器
def _create_callback_chain(
    callbacks: WorkflowCallbacks | None, progress: ProgressReporter | None
) -> WorkflowCallbacks:
    """Create a callbacks manager."""
    # 实例化一个 WorkflowCallbacksManager 对象
    manager = WorkflowCallbacksManager()
    # 如果有指定 callbacks，则注册到 manager 中
    if callbacks is not None:
        manager.register(callbacks)
    # 如果有指定 progress，则注册一个 ProgressWorkflowCallbacks 对象到 manager 中
    if progress is not None:
        manager.register(ProgressWorkflowCallbacks(progress))
    # 返回管理器对象
    return manager


async def _save_profiler_stats(
    storage: PipelineStorage, workflow_name: str, profile: MemoryProfile
):
    """Save the profiler stats to the storage."""
    # 将峰值统计数据保存到存储中
    await storage.set(
        f"{workflow_name}_profiling.peak_stats.csv",
        profile.peak_stats.to_csv(index=True),
    )

    # 将快照统计数据保存到存储中
    await storage.set(
        f"{workflow_name}_profiling.snapshot_stats.csv",
        profile.snapshot_stats.to_csv(index=True),
    )

    # 将时间统计数据保存到存储中
    await storage.set(
        f"{workflow_name}_profiling.time_stats.csv",
        profile.time_stats.to_csv(index=True),
    )

    # 将详细视图数据保存到存储中
    await storage.set(
        f"{workflow_name}_profiling.detailed_view.csv",
        profile.detailed_view.to_csv(index=True),
    )


async def _run_post_process_steps(
    post_process: list[PipelineWorkflowStep] | None,
    dataset: pd.DataFrame,
    context: PipelineRunContext,
    callbacks: WorkflowCallbacks,
) -> pd.DataFrame:
    """Run the pipeline.

    Args:
        - post_process - The post process steps to run
        - dataset - The dataset to run the steps on
        - context - The pipeline run context
    Returns:
        - output - The dataset after running the post process steps
    """
    # 如果有定义后处理步骤并且不为空
    if post_process is not None and len(post_process) > 0:
        # 创建一个名为 "Input Post Process" 的工作流，包含 post_process 中的步骤
        input_workflow = create_workflow(
            "Input Post Process",
            post_process,
        )
        # 将 dataset 添加为默认输入表格
        input_workflow.add_table(DEFAULT_INPUT_NAME, dataset)
        # 运行这个工作流
        await input_workflow.run(
            context=context,
            callbacks=callbacks,
        )
        # 将工作流的输出转换为 pandas DataFrame 类型，并赋值给 dataset
        dataset = cast(pd.DataFrame, input_workflow.output())
    # 返回处理后的 dataset
    return dataset


def _validate_dataset(dataset: pd.DataFrame):
    """Validate the dataset for the pipeline.

    Args:
        - dataset - The dataset to validate
    """
    # 如果 dataset 不是 pandas DataFrame 类型，抛出类型错误异常
    if not isinstance(dataset, pd.DataFrame):
        msg = "Dataset must be a pandas dataframe!"
        raise TypeError(msg)


def _apply_substitutions(config: PipelineConfig, run_id: str) -> PipelineConfig:
    # 定义替换规则，将 "timestamp" 替换为 run_id
    substitutions = {"timestamp": run_id}

    # 如果 config.storage 是 PipelineFileStorageConfig 或 PipelineBlobStorageConfig 类型，并且有指定 base_dir
    if (
        isinstance(
            config.storage, PipelineFileStorageConfig | PipelineBlobStorageConfig
        )
        and config.storage.base_dir
    ):
        # 使用 substitutions 对 base_dir 中的模板进行替换
        config.storage.base_dir = Template(config.storage.base_dir).substitute(
            substitutions
        )

    # 如果 config.cache 是 PipelineFileCacheConfig 或 PipelineBlobCacheConfig 类型，并且有指定 base_dir
    if (
        isinstance(config.cache, PipelineFileCacheConfig | PipelineBlobCacheConfig)
        and config.cache.base_dir
    ):
        # 使用 substitutions 对 base_dir 中的模板进行替换
        config.cache.base_dir = Template(config.cache.base_dir).substitute(
            substitutions
        )
    # 如果 config.reporting 是 PipelineFileReportingConfig 或 PipelineBlobReportingConfig 的实例，
    # 并且 config.reporting.base_dir 不为空，则执行以下操作：
    if (
        isinstance(
            config.reporting, PipelineFileReportingConfig | PipelineBlobReportingConfig
        )
        and config.reporting.base_dir
    ):
        # 使用模板替换 config.reporting.base_dir 中的变量
        config.reporting.base_dir = Template(config.reporting.base_dir).substitute(
            substitutions
        )
    
    # 返回修改后的 config 对象
    return config
# 定义一个函数 `_create_run_context`，用于创建流水线的运行上下文
def _create_run_context(
    # 参数：存储对象，用于存储流水线中间结果
    storage: PipelineStorage,
    # 参数：缓存对象，用于缓存流水线运行过程中的数据
    cache: PipelineCache,
    # 参数：运行统计对象，用于记录流水线的运行统计信息
    stats: PipelineRunStats,
) -> PipelineRunContext:
    """Create the run context for the pipeline."""
    # 创建并返回一个流水线运行上下文对象，包括统计信息、缓存和存储对象
    return PipelineRunContext(
        stats=stats,
        cache=cache,
        storage=storage,
    )
```