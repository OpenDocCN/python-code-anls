# `.\graphrag\graphrag\query\cli.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Command line interface for the query module."""

# 导入所需的库和模块
import os
from pathlib import Path
from typing import cast

import pandas as pd

# 导入自定义模块和函数
from graphrag.config import (
    GraphRagConfig,
    create_graphrag_config,
)
from graphrag.index.progress import PrintProgressReporter
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.vector_stores import VectorStoreFactory, VectorStoreType

# 导入本地模块
from .factories import get_global_search_engine, get_local_search_engine
from .indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)

# 初始化进度报告器
reporter = PrintProgressReporter("")

# 函数：获取嵌入描述存储
def __get_embedding_description_store(
    vector_store_type: str = VectorStoreType.LanceDB, config_args: dict | None = None
):
    """Get the embedding description store."""
    # 如果没有配置参数，则设为空字典
    if not config_args:
        config_args = {}

    # 更新配置参数，设置集合名称为配置中的查询集合名称或默认值"description_embedding"
    config_args.update({
        "collection_name": config_args.get(
            "query_collection_name",
            config_args.get("collection_name", "description_embedding"),
        ),
    })

    # 获取嵌入描述存储，使用工厂方法根据存储类型和配置参数获取存储对象
    description_embedding_store = VectorStoreFactory.get_vector_store(
        vector_store_type=vector_store_type, kwargs=config_args
    )

    # 连接到描述嵌入存储
    description_embedding_store.connect(**config_args)
    return description_embedding_store


# 函数：运行全局搜索
def run_global_search(
    data_dir: str | None,
    root_dir: str | None,
    community_level: int,
    response_type: str,
    query: str,
):
    """Run a global search with the given query."""
    # 配置路径和设置
    data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir)
    data_path = Path(data_dir)

    # 读取最终节点数据
    final_nodes: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_nodes.parquet"
    )
    # 读取最终实体数据
    final_entities: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_entities.parquet"
    )
    # 读取最终社区报告数据
    final_community_reports: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_community_reports.parquet"
    )

    # 读取索引报告数据
    reports = read_indexer_reports(
        final_community_reports, final_nodes, community_level
    )
    # 读取索引实体数据
    entities = read_indexer_entities(final_nodes, final_entities, community_level)
    # 获取全局搜索引擎
    search_engine = get_global_search_engine(
        config,
        reports=reports,
        entities=entities,
        response_type=response_type,
    )

    # 运行搜索并获取结果
    result = search_engine.search(query=query)

    # 打印全局搜索成功信息及响应结果
    reporter.success(f"Global Search Response: {result.response}")
    return result.response


# 函数：运行本地搜索
def run_local_search(
    data_dir: str | None,
    root_dir: str | None,
    community_level: int,
    response_type: str,
    query: str,
):
    """Run a local search with the given query."""
    # 配置路径和设置
    data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir)
    data_path = Path(data_dir)

    # 读取最终节点数据
    final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
    # 读取并加载最终社区报告的Parquet文件，存储为DataFrame
    final_community_reports = pd.read_parquet(
        data_path / "create_final_community_reports.parquet"
    )
    # 读取并加载最终文本单元的Parquet文件，存储为DataFrame
    final_text_units = pd.read_parquet(data_path / "create_final_text_units.parquet")
    # 读取并加载最终关系的Parquet文件，存储为DataFrame
    final_relationships = pd.read_parquet(
        data_path / "create_final_relationships.parquet"
    )
    # 读取并加载最终节点的Parquet文件，存储为DataFrame
    final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
    # 读取并加载最终实体的Parquet文件，存储为DataFrame
    final_entities = pd.read_parquet(data_path / "create_final_entities.parquet")
    # 构建最终协变量Parquet文件的路径
    final_covariates_path = data_path / "create_final_covariates.parquet"
    # 如果最终协变量Parquet文件存在，则加载为DataFrame；否则为None
    final_covariates = (
        pd.read_parquet(final_covariates_path)
        if final_covariates_path.exists()
        else None
    )

    # 获取嵌入向量存储的配置参数，如果未配置，则使用默认空字典
    vector_store_args = (
        config.embeddings.vector_store if config.embeddings.vector_store else {}
    )
    # 获取嵌入向量存储的类型，如果未配置则使用默认的LanceDB类型
    vector_store_type = vector_store_args.get("type", VectorStoreType.LanceDB)

    # 获取描述嵌入存储对象，根据给定的嵌入向量类型和配置参数
    description_embedding_store = __get_embedding_description_store(
        vector_store_type=vector_store_type,
        config_args=vector_store_args,
    )
    # 从最终节点和最终实体数据中读取索引器实体信息，返回实体列表
    entities = read_indexer_entities(final_nodes, final_entities, community_level)
    # 存储实体的语义嵌入向量
    store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store
    )
    # 读取最终协变量数据，如果数据存在则返回协变量列表，否则返回空列表
    covariates = (
        read_indexer_covariates(final_covariates)
        if final_covariates is not None
        else []
    )

    # 获取本地搜索引擎对象，配置包括报告、文本单元、实体、关系和协变量信息
    search_engine = get_local_search_engine(
        config,
        reports=read_indexer_reports(
            final_community_reports, final_nodes, community_level
        ),
        text_units=read_indexer_text_units(final_text_units),
        entities=entities,
        relationships=read_indexer_relationships(final_relationships),
        covariates={"claims": covariates},
        description_embedding_store=description_embedding_store,
        response_type=response_type,
    )

    # 使用搜索引擎执行查询
    result = search_engine.search(query=query)
    # 输出成功的本地搜索响应信息
    reporter.success(f"Local Search Response: {result.response}")
    # 返回搜索结果响应
    return result.response
# 配置路径和设置函数，接收数据目录和根目录作为参数，返回三元组(data_dir, root_dir, config)
def _configure_paths_and_settings(
    data_dir: str | None, root_dir: str | None
) -> tuple[str, str | None, GraphRagConfig]:
    # 如果data_dir和root_dir都为None，则抛出数值错误
    if data_dir is None and root_dir is None:
        msg = "Either data_dir or root_dir must be provided."
        raise ValueError(msg)
    
    # 如果data_dir为None，则通过root_dir推断data_dir
    if data_dir is None:
        data_dir = _infer_data_dir(cast(str, root_dir))
    
    # 创建GraphRag配置对象
    config = _create_graphrag_config(root_dir, data_dir)
    return data_dir, root_dir, config


# 推断数据目录函数，接收根目录作为参数，返回推断的数据目录路径字符串
def _infer_data_dir(root: str) -> str:
    # 创建output目录路径
    output = Path(root) / "output"
    # 如果output目录存在
    if output.exists():
        # 获取output目录下所有文件夹，并按修改时间降序排序
        folders = sorted(output.iterdir(), key=os.path.getmtime, reverse=True)
        # 如果文件夹数量大于0
        if len(folders) > 0:
            # 取最新的文件夹
            folder = folders[0]
            # 返回artifacts文件夹的绝对路径作为数据目录
            return str((folder / "artifacts").absolute())
    
    # 如果无法推断数据目录，则抛出数值错误
    msg = f"Could not infer data directory from root={root}"
    raise ValueError(msg)


# 创建GraphRag配置对象函数，接收根目录和数据目录作为参数，返回GraphRagConfig对象
def _create_graphrag_config(root: str | None, data_dir: str | None) -> GraphRagConfig:
    """Create a GraphRag configuration."""
    # 调用_read_config_parameters函数，传入root或data_dir参数并返回结果
    return _read_config_parameters(cast(str, root or data_dir))


# 读取配置参数函数，接收根目录作为参数，尝试从文件或环境变量中读取配置并返回配置对象
def _read_config_parameters(root: str):
    # 将root转换为Path对象
    _root = Path(root)
    # 设置settings.yaml文件路径
    settings_yaml = _root / "settings.yaml"
    # 如果settings.yaml文件不存在，则尝试使用settings.yml文件
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"
    # 设置settings.json文件路径
    settings_json = _root / "settings.json"

    # 如果settings.yaml文件存在
    if settings_yaml.exists():
        # 记录信息，从settings.yaml文件中读取设置
        reporter.info(f"Reading settings from {settings_yaml}")
        # 打开settings.yaml文件，并安全加载yaml数据
        with settings_yaml.open("r") as file:
            import yaml
            data = yaml.safe_load(file)
            # 调用create_graphrag_config函数，传入数据和根目录参数，并返回配置对象
            return create_graphrag_config(data, root)

    # 如果settings.json文件存在
    if settings_json.exists():
        # 记录信息，从settings.json文件中读取设置
        reporter.info(f"Reading settings from {settings_json}")
        # 打开settings.json文件，并加载其中的json数据
        with settings_json.open("r") as file:
            import json
            data = json.loads(file.read())
            # 调用create_graphrag_config函数，传入数据和根目录参数，并返回配置对象
            return create_graphrag_config(data, root)

    # 如果以上文件都不存在，则记录信息，从环境变量中读取设置
    reporter.info("Reading settings from environment variables")
    # 调用create_graphrag_config函数，传入root_dir参数，并返回配置对象
    return create_graphrag_config(root_dir=root)
```