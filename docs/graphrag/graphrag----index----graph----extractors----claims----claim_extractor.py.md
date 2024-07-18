# `.\graphrag\graphrag\index\graph\extractors\claims\claim_extractor.py`

```py
    llm_invoker: CompletionLLM,
    extraction_prompt: str | None = None,
    input_text_key: str | None = None,
    input_entity_spec_key: str | None = None,
    input_claim_description_key: str | None = None,
    input_resolved_entities_key: str | None = None,
    tuple_delimiter_key: str | None = None,
    record_delimiter_key: str | None = None,
    completion_delimiter_key: str | None = None,
    encoding_model: str | None = None,
    max_gleanings: int | None = None,
    on_error: ErrorHandlerFn | None = None,
):
    """
    Initialize the ClaimExtractor class.

    Args:
        llm_invoker: Instance of CompletionLLM used for invoking language models.
        extraction_prompt: Optional prompt for extraction.
        input_text_key: Optional key for input text.
        input_entity_spec_key: Optional key for entity specification in input.
        input_claim_description_key: Optional key for claim description in input.
        input_resolved_entities_key: Optional key for resolved entities in input.
        tuple_delimiter_key: Optional key for tuple delimiter.
        record_delimiter_key: Optional key for record delimiter.
        completion_delimiter_key: Optional key for completion delimiter.
        encoding_model: Optional encoding model specification.
        max_gleanings: Optional maximum number of gleanings to process.
        on_error: Optional error handler function.
    """
    # Assigning provided parameters to corresponding class attributes
    self._llm = llm_invoker
    self._extraction_prompt = extraction_prompt if extraction_prompt is not None else CLAIM_EXTRACTION_PROMPT
    self._summary_prompt = CONTINUE_PROMPT
    self._output_formatter_prompt = LOOP_PROMPT
    self._input_text_key = input_text_key if input_text_key is not None else defs.INPUT_TEXT_KEY
    self._input_entity_spec_key = input_entity_spec_key if input_entity_spec_key is not None else defs.INPUT_ENTITY_SPEC_KEY
    self._input_claim_description_key = input_claim_description_key if input_claim_description_key is not None else defs.INPUT_CLAIM_DESCRIPTION_KEY
    self._tuple_delimiter_key = tuple_delimiter_key if tuple_delimiter_key is not None else DEFAULT_TUPLE_DELIMITER
    self._record_delimiter_key = record_delimiter_key if record_delimiter_key is not None else DEFAULT_RECORD_DELIMITER
    self._completion_delimiter_key = completion_delimiter_key if completion_delimiter_key is not None else DEFAULT_COMPLETION_DELIMITER
    self._max_gleanings = max_gleanings if max_gleanings is not None else defs.DEFAULT_MAX_GLEANINGS
    self._on_error = on_error if on_error is not None else defs.DEFAULT_ERROR_HANDLER

    # Logging initialization for this class
    log.info("ClaimExtractor instance initialized.")
        """Init method definition."""
        # 初始化方法的定义，用于实例化对象时设置初始参数
        self._llm = llm_invoker
        # 设置语言模型调用器实例的私有属性 _llm
        self._extraction_prompt = extraction_prompt or CLAIM_EXTRACTION_PROMPT
        # 设置提取提示文本的私有属性 _extraction_prompt，如果 extraction_prompt 为 None，则使用默认值 CLAIM_EXTRACTION_PROMPT
        self._input_text_key = input_text_key or "input_text"
        # 设置输入文本键的私有属性 _input_text_key，如果 input_text_key 为 None，则使用默认值 "input_text"
        self._input_entity_spec_key = input_entity_spec_key or "entity_specs"
        # 设置实体规格键的私有属性 _input_entity_spec_key，如果 input_entity_spec_key 为 None，则使用默认值 "entity_specs"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        # 设置元组分隔键的私有属性 _tuple_delimiter_key，如果 tuple_delimiter_key 为 None，则使用默认值 "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        # 设置记录分隔键的私有属性 _record_delimiter_key，如果 record_delimiter_key 为 None，则使用默认值 "record_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        # 设置完成分隔键的私有属性 _completion_delimiter_key，如果 completion_delimiter_key 为 None，则使用默认值 "completion_delimiter"
        self._input_claim_description_key = (
            input_claim_description_key or "claim_description"
        )
        # 设置输入声明描述键的私有属性 _input_claim_description_key，如果 input_claim_description_key 为 None，则使用默认值 "claim_description"
        self._input_resolved_entities_key = (
            input_resolved_entities_key or "resolved_entities"
        )
        # 设置输入已解析实体键的私有属性 _input_resolved_entities_key，如果 input_resolved_entities_key 为 None，则使用默认值 "resolved_entities"
        self._max_gleanings = (
            max_gleanings if max_gleanings is not None else defs.CLAIM_MAX_GLEANINGS
        )
        # 设置最大获取数的私有属性 _max_gleanings，如果 max_gleanings 不为 None，则使用其值，否则使用默认值 defs.CLAIM_MAX_GLEANINGS
        self._on_error = on_error or (lambda _e, _s, _d: None)
        # 设置错误处理回调函数的私有属性 _on_error，如果 on_error 不为 None，则使用其值，否则使用一个空函数

        # Construct the looping arguments
        # 构造循环参数
        encoding = tiktoken.get_encoding(encoding_model or "cl100k_base")
        # 使用指定的编码模型或默认的 "cl100k_base" 获取编码对象
        yes = encoding.encode("YES")
        # 使用编码对象对字符串 "YES" 进行编码
        no = encoding.encode("NO")
        # 使用编码对象对字符串 "NO" 进行编码
        self._loop_args = {"logit_bias": {yes[0]: 100, no[0]: 100}, "max_tokens": 1}
        # 设置循环参数的私有属性 _loop_args，包括对 "YES" 和 "NO" 编码后的偏置和最大令牌数
    ) -> ClaimExtractorResult:
        """定义方法调用。"""
        # 如果未提python
    ) -> ClaimExtractorResult:
        """Method definition for calling the extractor."""
        
        # Initialize prompt_variables if not provided
        if prompt_variables is None:
            prompt_variables = {}
        
        # Extract necessary inputs
        texts = inputs[self._input_text_key]
        entity_spec = str(inputs[self._input_entity_spec_key])
        claim_description = inputs[self._input_claim_description_key]
        resolved_entities = inputs.get(self._input_resolved_entities_key, {})
        source_doc_map = {}

        # Set up prompt_args with default delimiters if not provided in prompt_variables
        prompt_args = {
            self._input_entity_spec_key: entity_spec,
            self._input_claim_description_key: claim_description,
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
            or DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
            or DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: prompt_variables.get(
                self._completion_delimiter_key
            )
            or DEFAULT_COMPLETION_DELIMITER,
        }

        # Initialize list to hold all extracted claims
        all_claims: list[dict] = []

        # Process each document in texts
        for doc_index, text in enumerate(texts):
            document_id = f"d{doc_index}"
            try:
                # Extract claims asynchronously from the document
                claims = await self._process_document(prompt_args, text, doc_index)
                
                # Clean and store each claim
                all_claims += [
                    self._clean_claim(c, document_id, resolved_entities) for c in claims
                ]
                
                # Map document_id to its original text for reference
                source_doc_map[document_id] = text
                
            except Exception as e:
                # Log and handle errors during claim extraction
                log.exception("error extracting claim")
                self._on_error(
                    e,
                    traceback.format_exc(),
                    {"doc_index": doc_index, "text": text},
                )
                continue

        # Return extracted claims and source document mapping
        return ClaimExtractorResult(
            output=all_claims,
            source_docs=source_doc_map,
        )

    def _clean_claim(
        self, claim: dict, document_id: str, resolved_entities: dict
    ) -> dict:
        """Clean up extracted claim by replacing subject and object with resolved entities if available."""
        
        # Retrieve subject and object IDs from claim dictionary
        obj = claim.get("object_id", claim.get("object"))
        subject = claim.get("subject_id", claim.get("subject"))

        # Replace subject and object IDs with resolved entities if available
        obj = resolved_entities.get(obj, obj)
        subject = resolved_entities.get(subject, subject)
        
        # Update claim dictionary with cleaned IDs and document ID
        claim["object_id"] = obj
        claim["subject_id"] = subject
        claim["doc_id"] = document_id
        
        return claim

    async def _process_document(
        self, prompt_args: dict, doc, doc_index: int
    ):
        """Async method to process a document and extract claims based on provided arguments."""
    ) -> list[dict]:
        # 从prompt_args中获取记录分隔符，如果没有则使用默认值
        record_delimiter = prompt_args.get(
            self._record_delimiter_key, DEFAULT_RECORD_DELIMITER
        )
        # 从prompt_args中获取完成分隔符，如果没有则使用默认值
        completion_delimiter = prompt_args.get(
            self._completion_delimiter_key, DEFAULT_COMPLETION_DELIMITER
        )

        # 使用LLM模型进行异步调用，传递doc和prompt_args作为变量
        response = await self._llm(
            self._extraction_prompt,
            variables={
                self._input_text_key: doc,
                **prompt_args,
            },
        )
        # 获取LLM模型的输出结果，如果为None则赋空字符串
        results = response.output or ""
        # 去除结果的首尾空白字符，并去除末尾的完成分隔符
        claims = results.strip().removesuffix(completion_delimiter)

        # 重复以确保最大化实体数量
        for i in range(self._max_gleanings):
            # 使用LLM模型进行异步调用，继续提取，传递history作为变量
            glean_response = await self._llm(
                CONTINUE_PROMPT,
                name=f"extract-continuation-{i}",
                history=response.history or [],
            )
            # 获取继续提取的扩展输出，如果为None则赋空字符串
            extension = glean_response.output or ""
            # 将扩展输出去除首尾空白字符，并去除末尾的完成分隔符，添加到claims中
            claims += record_delimiter + extension.strip().removesuffix(
                completion_delimiter
            )

            # 如果不是最后一次循环，检查是否应继续
            if i >= self._max_gleanings - 1:
                break

            # 使用LLM模型进行异步调用，检查是否应继续，传递model_parameters作为变量
            continue_response = await self._llm(
                LOOP_PROMPT,
                name=f"extract-loopcheck-{i}",
                history=glean_response.history or [],
                model_parameters=self._loop_args,
            )
            # 如果继续的回应不是"YES"，则跳出循环
            if continue_response.output != "YES":
                break

        # 解析提取的结果，返回结果列表
        result = self._parse_claim_tuples(results, prompt_args)
        # 为每个结果项添加doc_id字段，使用doc_index作为标识
        for r in result:
            r["doc_id"] = f"{doc_index}"
        return result

    # 解析声明元组的私有方法
    def _parse_claim_tuples(
        self, claims: str, prompt_variables: dict
        ) -> list[dict[str, Any]]:
        """Parse claim tuples."""
        # 获取记录分隔符，如果未设置则使用默认分隔符
        record_delimiter = prompt_variables.get(
            self._record_delimiter_key, DEFAULT_RECORD_DELIMITER
        )
        # 获取完成分隔符，如果未设置则使用默认分隔符
        completion_delimiter = prompt_variables.get(
            self._completion_delimiter_key, DEFAULT_COMPLETION_DELIMITER
        )
        # 获取元组分隔符，如果未设置则使用默认分隔符
        tuple_delimiter = prompt_variables.get(
            self._tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER
        )

        def pull_field(index: int, fields: list[str]) -> str | None:
            """提取字段内容或返回None。"""
            return fields[index].strip() if len(fields) > index else None

        # 初始化结果列表
        result: list[dict[str, Any]] = []
        # 清理并分割声明字符串为单独的声明值列表
        claims_values = (
            claims.strip().removesuffix(completion_delimiter).split(record_delimiter)
        )
        # 遍历每个声明值
        for claim in claims_values:
            # 清理声明值并去除括号
            claim = claim.strip().removeprefix("(").removesuffix(")")

            # 忽略完成分隔符
            if claim == completion_delimiter:
                continue

            # 拆分声明值为字段列表
            claim_fields = claim.split(tuple_delimiter)
            # 将字段映射到字典并添加到结果列表中
            result.append({
                "subject_id": pull_field(0, claim_fields),
                "object_id": pull_field(1, claim_fields),
                "type": pull_field(2, claim_fields),
                "status": pull_field(3, claim_fields),
                "start_date": pull_field(4, claim_fields),
                "end_date": pull_field(5, claim_fields),
                "description": pull_field(6, claim_fields),
                "source_text": pull_field(7, claim_fields),
                "doc_id": pull_field(8, claim_fields),
            })
        # 返回最终的结果列表
        return result
```