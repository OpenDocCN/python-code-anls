# `.\MinerU\tests\test_cli\conf\conf.py`

```
# 导入操作系统相关的模块
import os
# 创建一个字典用于存储配置项
conf = {
    # 从环境变量获取 GITHUB_WORKSPACE 的值，作为代码路径
    "code_path": os.environ.get('GITHUB_WORKSPACE'),
    # 生成 PDF 开发路径，拼接代码路径与子路径
    "pdf_dev_path" : os.environ.get('GITHUB_WORKSPACE') + "/tests/test_cli/pdf_dev",
    # 设置 PDF 结果存储路径
    "pdf_res_path": "/tmp/magic-pdf"
}
```