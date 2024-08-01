# `.\DB-GPT-src\examples\client\knowledge_crud_example.py`

```py
# 异步主函数，用于演示如何使用 dbgpt 客户端进行知识空间和文档的创建、获取、更新和删除操作
async def main():
    # 初始化客户端，使用给定的 API 密钥
    DBGPT_API_KEY = "dbgpt"
    client = Client(api_key=DBGPT_API_KEY)

    # 创建一个新的空间
    res = await create_space(
        client,
        SpaceModel(
            name="test_space_1",
            vector_type="Chroma",
            desc="for client space desc",
            owner="dbgpt",
        ),
    )
    print(res)

    # 下面是一些注释掉的操作示例，分别展示了如何列出所有空间、获取空间、创建空间、更新空间、删除空间等操作
    # 列出所有空间
    # res = await list_space(client)
    # print(res)

    # 获取空间
    # res = await get_space(client, space_id='5')

    # 创建空间
    # res = await create_space(client, SpaceModel(name="test_space", vector_type="Chroma", desc="for client space", owner="dbgpt"))

    # 更新空间
    # res = await update_space(client, SpaceModel(name="test_space", vector_type="Chroma", desc="for client space333", owner="dbgpt"))

    # 删除空间
    # res = await delete_space(client, space_id='31')
    # print(res)

    # 列出所有文档
    # 调用异步函数 `list_document` 获取文档列表的结果并赋值给 `res`
    # res = await list_document(client)
    
    # 调用异步函数 `get_document` 根据文档ID（"52"）获取特定文档的结果并赋值给 `res`
    # res = await get_document(client, "52")
    
    # 调用异步函数 `delete_document` 删除指定文档（ID为 "150"）的结果并赋值给 `res`
    # res = await delete_document(client, "150")
    
    # 调用异步函数 `create_document` 创建文档的结果并赋值给 `res`，传入了文档相关的信息，包括文件路径和内容等
    # res = await create_document(client, DocumentModel(space_id="5", doc_name="test_doc", doc_type="test", doc_content="test content"
    #                                                   , doc_file=('your_file_name', open('{your_file_path}', 'rb'))))
    
    # 调用异步函数 `sync_document` 同步文档的结果并赋值给 `res`，传入了同步模型相关的参数
    # res = await sync_document(client, sync_model=SyncModel(doc_id="157", space_id="49", model_name="text2vec", chunk_parameters=ChunkParameters(chunk_strategy="Automatic")))
# 如果当前脚本作为主程序执行（而不是作为模块被导入），则执行以下代码块
if __name__ == "__main__":
    # 使用 asyncio 模块运行 main 函数，这会启动异步事件循环并执行 main 函数
    asyncio.run(main())
```