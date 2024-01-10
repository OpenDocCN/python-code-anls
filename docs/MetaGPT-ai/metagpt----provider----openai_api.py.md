# `MetaGPT\metagpt\provider\openai_api.py`

```

# 设置文件编码为 utf-8
# 定义文件的作者、时间和名称
# 修改记录：mashenquan, 2023/8/20。移除全局配置 `CONFIG`，启用隔离的配置支持；将成本控制从全局改为公司级别。
# 修改记录：mashenquan, 2023/11/21。修复 bug：ReadTimeout。
# 修改记录：mashenquan, 2023/12/1。修复 bug：openai 0.x 导致的未关闭连接。
# 导入所需的模块
# 定义异步迭代器和联合类型
# 导入 openai 模块中的异常和类
# 导入 openai 模块中的类型
# 导入 tenacity 模块中的装饰器和函数
# 导入 metagpt 模块中的配置和日志
# 导入 metagpt 模块中的提供者基类和常量
# 导入 metagpt 模块中的提供者注册表
# 导入 metagpt 模块中的消息模式和工具选择
# 导入 metagpt 模块中的成本管理器和异常处理函数
# 导入 metagpt 模块中的令牌计数器函数
# 定义日志和重新引发异常的函数
# 注册 LLMProviderEnum.OPENAI 提供者

```