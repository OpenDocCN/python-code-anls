# `.\graphrag\graphrag\prompt_tune\cli.py`

```py
"""Command line interface for the fine_tune module."""

# 导入所需的库
from pathlib import Path
from datashaper import NoopVerbCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.llm import load_llm
from graphrag.index.progress import PrintProgressReporter
from graphrag.index.progress.types import ProgressReporter
from graphrag.llm.types.llm_types import CompletionLLM
from graphrag.prompt_tune.generator import (
    MAX_TOKEN_COUNT,
    create_community_summarization_prompt,
    create_entity_extraction_prompt,
    create_entity_summarization_prompt,
    detect_language,
    generate_community_report_rating,
    generate_community_reporter_role,
    generate_domain,
    generate_entity_relationship_examples,
    generate_entity_types,
    generate_persona,
)
from graphrag.prompt_tune.loader import (
    MIN_CHUNK_SIZE,
    load_docs_in_chunks,
    read_config_parameters,
)

# 定义函数fine_tune，用于微调模型
async def fine_tune(
    root: str,
    domain: str,
    select: str = "random",
    limit: int = 15,
    max_tokens: int = MAX_TOKEN_COUNT,
    chunk_size: int = MIN_CHUNK_SIZE,
    language: str | None = None,
    skip_entity_types: bool = False,
    output: str = "prompts",
):
    """Fine tune the model.

    Parameters
    ----------
    - root: The root directory.
    - domain: The domain to map the input documents to.
    - select: The chunk selection method.
    - limit: The limit of chunks to load.
    - max_tokens: The maximum number of tokens to use on entity extraction prompts.
    - chunk_size: The chunk token size to use.
    - skip_entity_types: Skip generating entity types.
    - output: The output folder to store the prompts.
    """
    # 创建一个打印进度的报告实例
    reporter = PrintProgressReporter("")
    # 从根目录中读取配置参数
    config = read_config_parameters(root, reporter)
    
    # 调用fine_tune_with_config函数，传入参数
    await fine_tune_with_config(
        root,
        config,
        domain,
        select,
        limit,
        max_tokens,
        chunk_size,
        language,
        skip_entity_types,
        output,
        reporter,
    )

# 定义函数fine_tune_with_config，用于根据配置微调模型
async def fine_tune_with_config(
    root: str,
    config: GraphRagConfig,
    domain: str,
    select: str = "random",
    limit: int = 15,
    max_tokens: int = MAX_TOKEN_COUNT,
    chunk_size: int = MIN_CHUNK_SIZE,
    language: str | None = None,
    skip_entity_types: bool = False,
    output: str = "prompts",
    reporter: ProgressReporter | None = None,
):
    """Fine tune the model with a configuration.

    Parameters
    ----------
    - root: The root directory.
    - config: The GraphRag configuration.
    - domain: The domain to map the input documents to.
    - select: The chunk selection method.
    - limit: The limit of chunks to load.
    - max_tokens: The maximum number of tokens to use on entity extraction prompts.
    - chunk_size: The chunk token size to use for input text units.
    - skip_entity_types: Skip generating entity types.
    """
    - output: 存储生成提示的输出文件夹路径。
    - reporter: 进度报告器对象，用于跟踪操作进度。

    Returns
    -------
    - None
    """
    如果没有提供进度报告器对象，就创建一个默认的 PrintProgressReporter("")。

    将输出路径设定为根目录和指定的输出文件夹名称的组合。
    output_path = Path(config.root_dir) / output

    使用异步方式分批加载文档数据，并返回文档列表。
    doc_list = await load_docs_in_chunks(
        root=root,
        config=config,
        limit=limit,
        select_method=select,
        reporter=reporter,
        chunk_size=chunk_size,
    )

    根据配置信息加载语言模型（LLM）。
    llm = load_llm(
        "fine_tuning",
        config.llm.type,
        NoopVerbCallbacks(),
        None,
        config.llm.model_dump(),
    )

    使用加载的语言模型生成索引提示。
    await generate_indexing_prompts(
        llm,
        config,
        doc_list,
        output_path,
        reporter,
        domain,
        language,
        max_tokens,
        skip_entity_types,
    )
async def generate_indexing_prompts(
    llm: CompletionLLM,
    config: GraphRagConfig,
    doc_list: list[str],
    output_path: Path,
    reporter: ProgressReporter,
    domain: str | None = None,
    language: str | None = None,
    max_tokens: int = MAX_TOKEN_COUNT,
    skip_entity_types: bool = False,
):
    """Generate indexing prompts.

    Parameters
    ----------
    - llm: The LLM model to use.
    - config: The GraphRag configuration.
    - doc_list: The list of documents to use.
    - output_path: The path to store the prompts.
    - reporter: The progress reporter.
    - domain: The domain to map the input documents to.
    - max_tokens: The maximum number of tokens to use on entity extraction prompts
    - skip_entity_types: Skip generating entity types.
    """
    # 如果没有指定 domain，生成领域信息
    if not domain:
        reporter.info("Generating domain...")
        domain = await generate_domain(llm, doc_list)
        reporter.info(f"Generated domain: {domain}")

    # 如果没有指定 language，检测文档语言
    if not language:
        reporter.info("Detecting language...")
        language = await detect_language(llm, doc_list)
        reporter.info(f"Detected language: {language}")

    # 生成 persona 信息
    reporter.info("Generating persona...")
    persona = await generate_persona(llm, domain)
    reporter.info(f"Generated persona: {persona}")

    # 生成社区报告排名描述
    reporter.info("Generating community report ranking description...")
    community_report_ranking = await generate_community_report_rating(
        llm, domain=domain, persona=persona, docs=doc_list
    )
    reporter.info(
        f"Generated community report ranking description: {community_report_ranking}"
    )

    # 如果不跳过生成实体类型，则生成实体类型信息
    entity_types = None
    if not skip_entity_types:
        reporter.info("Generating entity types")
        entity_types = await generate_entity_types(
            llm,
            domain=domain,
            persona=persona,
            docs=doc_list,
            json_mode=config.llm.model_supports_json or False,
        )
        reporter.info(f"Generated entity types: {entity_types}")

    # 生成实体关系示例
    reporter.info("Generating entity relationship examples...")
    examples = await generate_entity_relationship_examples(
        llm,
        persona=persona,
        entity_types=entity_types,
        docs=doc_list,
        language=language,
        json_mode=False,  # config.llm.model_supports_json should be used, but this prompts are used in non-json by the index engine
    )
    reporter.info("Done generating entity relationship examples")

    # 生成实体抽取提示
    reporter.info("Generating entity extraction prompt...")
    create_entity_extraction_prompt(
        entity_types=entity_types,
        docs=doc_list,
        examples=examples,
        language=language,
        json_mode=False,  # config.llm.model_supports_json should be used, but this prompts are used in non-json by the index engine
        output_path=output_path,
        encoding_model=config.encoding_model,
        max_token_count=max_tokens,
    )
    # 输出信息，指示已生成实体提取提示，并指定存储位置
    reporter.info(f"Generated entity extraction prompt, stored in folder {output_path}")

    # 输出信息，指示正在生成实体摘要提示
    create_entity_summarization_prompt(
        # 传递个人信息
        persona=persona,
        # 传递语言信息
        language=language,
        # 指定输出路径
        output_path=output_path,
    )

    # 输出信息，指示已生成实体摘要提示，并指定存储位置
    reporter.info(
        f"Generated entity summarization prompt, stored in folder {output_path}"
    )

    # 输出信息，指示正在生成社区报告者角色
    community_reporter_role = await generate_community_reporter_role(
        # 传递语言模型
        llm,
        # 指定领域
        domain=domain,
        # 传递个人信息
        persona=persona,
        # 传递文档列表
        docs=doc_list
    )

    # 输出信息，指示已生成社区报告者角色
    reporter.info(f"Generated community reporter role: {community_reporter_role}")

    # 输出信息，指示正在生成社区摘要提示
    create_community_summarization_prompt(
        # 传递个人信息
        persona=persona,
        # 传递社区报告者角色
        role=community_reporter_role,
        # 传递社区报告排名描述
        report_rating_description=community_report_ranking,
        # 传递语言信息
        language=language,
        # 指定输出路径
        output_path=output_path,
    )

    # 输出信息，指示已生成社区摘要提示，并指定存储位置
    reporter.info(
        f"Generated community summarization prompt, stored in folder {output_path}"
    )
```