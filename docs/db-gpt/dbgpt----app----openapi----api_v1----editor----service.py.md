# `.\DB-GPT-src\dbgpt\app\openapi\api_v1\editor\service.py`

```py
    ) -> Union[Dict[str, Union[str, List[Union[str, Dict[str, Union[str, List[str]]]]]]], Result]:
        storage_conv: StorageConversation = self.get_storage_conv(conv_uid)
        messages_by_round = _split_messages_by_round(storage_conv.messages)
        result: Dict[str, Union[str, List[Union[str, Dict[str, Union[str, List[str]]]]]]] = {}
        for one_round_message in messages_by_round:
            if not one_round_message:
                continue
            for message in one_round_message:
                if message.type == "human" and message.round_index == round_index:
                    round_name = message.content
                    param_value = message.additional_kwargs.get("param_value")
                    sql_query = message.additional_kwargs.get("sql_query")
                    if param_value:
                        result["round_name"] = round_name
                        result["db_name"] = param_value
                        result["sql_query"] = sql_query
                        result["message"] = [message.content]
                        return result
                    else:
                        result["message"] = [message.content]
                        return result
    ) -> Optional[Union[List, Dict]]:
        # 获取指定对话 UID 的存储对话对象
        storage_conv: StorageConversation = self.get_storage_conv(conv_uid)
        # 将消息按照轮次分组
        messages_by_round = _split_messages_by_round(storage_conv.messages)
        # 遍历每一轮的消息列表
        for one_round_message in messages_by_round:
            # 如果消息列表为空，则跳过当前轮次的处理
            if not one_round_message:
                continue
            # 遍历当前轮次的每条消息
            for message in one_round_message:
                # 如果消息类型为 "ai" 并且轮次索引匹配输入的轮次索引
                if message.type == "ai" and message.round_index == round_index:
                    # 获取消息内容
                    content = message.content
                    # 记录日志，输出历史 AI 的 JSON 响应内容
                    logger.info(f"history ai json resp: {content}")
                    # 解析消息内容，转换为字典格式
                    context_dict = _parse_pure_dict(content)
                    # 返回解析后的消息内容字典
                    return context_dict
        # 如果未找到匹配的消息，返回空值
        return None

    def sql_editor_submit_and_save(
        self, sql_edit_context: ChatSqlEditContext, connection: BaseConnector
    ):
        # 获取指定对话 UID 的存储对话对象
        storage_conv: StorageConversation = self.get_storage_conv(
            sql_edit_context.conv_uid
        )
        # 如果不支持独立会话模式，抛出数值错误异常
        if not storage_conv.save_message_independent:
            raise ValueError(
                "Submit sql and save just support independent conversation mode(after v0.4.6)"
            )
        # 获取对话服务对象
        conv_serve: ConversationServe = self.conv_serve()
        # 将消息按照轮次分组
        messages_by_round = _split_messages_by_round(storage_conv.messages)
        # 初始化待更新的消息列表
        to_update_messages = []
        # 遍历每一轮的消息列表
        for one_round_message in messages_by_round:
            # 如果消息列表为空，则跳过当前轮次的处理
            if not one_round_message:
                continue
            # 如果当前轮次的第一条消息的轮次索引与输入的会话轮次索引匹配
            if one_round_message[0].round_index == sql_edit_context.conv_round:
                # 遍历当前轮次的每条消息
                for message in one_round_message:
                    # 如果消息类型为 "ai"
                    if message.type == "ai":
                        # 解析消息内容为字典格式
                        db_resp = _parse_pure_dict(message.content)
                        # 更新字典中的 "thoughts" 和 "sql" 字段
                        db_resp["thoughts"] = sql_edit_context.new_speak
                        db_resp["sql"] = sql_edit_context.new_sql
                        # 将更新后的字典转换为 JSON 字符串，并更新消息内容
                        message.content = json.dumps(db_resp, ensure_ascii=False)
                        # 创建消息存储项，并添加到待更新消息列表中
                        to_update_messages.append(
                            MessageStorageItem(
                                storage_conv.conv_uid, message.index, message.to_dict()
                            )
                        )
                    # 如果消息类型为 "view"，暂不支持更新视图消息
                    # if message.type == "view":
                    #     data_loader = DbDataLoader()
                    #     message.content = data_loader.get_table_view_by_conn(
                    #         connection.run_to_df(sql_edit_context.new_sql),
                    #         sql_edit_context.new_speak,
                    #     )
                    #     to_update_messages.append(
                    #         MessageStorageItem(
                    #             storage_conv.conv_uid, message.index, message.to_dict()
                    #         )
                    #     )
                # 如果有待更新的消息列表，则保存或更新这些消息
                if to_update_messages:
                    conv_serve.message_storage.save_or_update_list(to_update_messages)
                # 处理完当前轮次后返回
                return
    # 获取指定会话的存储对象
    storage_conv: StorageConversation = self.get_storage_conv(conv_uid)
    # 根据每轮消息拆分消息列表
    messages_by_round = _split_messages_by_round(storage_conv.messages)
    # 遍历每一轮的消息列表
    for one_round_message in messages_by_round:
        # 跳过空消息列表
        if not one_round_message:
            continue
        # 遍历每条消息
        for message in one_round_message:
            # 如果消息类型为 "ai"
            if message.type == "ai":
                # 解析消息内容为纯字典形式
                context_dict = _parse_pure_dict(message.content)
                # 创建图表列表对象
                chart_list: ChartList = ChartList(
                    round=message.round_index,
                    db_name=message.additional_kwargs.get("param_value"),
                    charts=context_dict,
                )
                return chart_list  # 返回图表列表对象作为结果

    # 如果没有找到符合条件的消息，则返回空（Optional[ChartList]）
    # 说明未找到满足条件的图表列表对象
    return None


    # 获取指定会话和图表标题的编辑器图表信息
    storage_conv: StorageConversation = self.get_storage_conv(conv_uid)
    # 根据每轮消息拆分消息列表
    messages_by_round = _split_messages_by_round(storage_conv.messages)
    # 遍历每一轮的消息列表
    for one_round_message in messages_by_round:
        # 跳过空消息列表
        if not one_round_message:
            continue
        # 遍历每条消息
        for message in one_round_message:
            # 获取消息中的数据库名称
            db_name = message.additional_kwargs.get("param_value")
            # 如果数据库名称为空
            if not db_name:
                # 记录错误日志并返回失败结果对象
                logger.error(
                    "this dashboard dialogue version too old, can't support editor!"
                )
                return Result.failed(
                    msg="this dashboard dialogue version too old, can't support editor!"
                )
            # 如果消息类型为 "view"
            if message.type == "view":
                # 解析消息内容为纯字典形式
                view_data: dict = _parse_pure_dict(message.content)
                # 获取视图数据中的图表列表
                charts: List = view_data.get("charts")
                # 查找符合指定图表标题的图表对象
                find_chart = list(
                    filter(lambda x: x["chart_name"] == chart_title, charts)
                )[0]

                # 获取本地数据库连接器
                conn = cfg.local_db_manager.get_connector(db_name)
                # 创建图表详细信息对象
                detail: ChartDetail = ChartDetail(
                    chart_uid=find_chart["chart_uid"],
                    chart_type=find_chart["chart_type"],
                    chart_desc=find_chart["chart_desc"],
                    chart_sql=find_chart["chart_sql"],
                    db_name=db_name,
                    chart_name=find_chart["chart_name"],
                    chart_value=find_chart["values"],
                    table_value=conn.run(find_chart["chart_sql"]),
                )
                return Result.succ(detail)  # 返回成功的图表详细信息对象

    # 如果没有找到符合条件的消息，则返回失败结果对象
    return Result.failed(msg="Can't Find Chart Detail Info!")
# 定义一个函数，用于解析纯文本格式的字符串，并返回解析后的结果，可以是字典或列表
def _parse_pure_dict(res_str: str) -> Union[Dict, List]:
    # 创建一个 BaseOutputParser 的实例对象
    output_parser = BaseOutputParser()
    # 使用 output_parser 对象解析 res_str，得到解析后的上下文信息
    context = output_parser.parse_prompt_response(res_str)
    # 将上下文信息解析为 JSON 格式，并返回解析后的结果
    return json.loads(context)
```