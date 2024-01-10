# `MetaGPT\tests\metagpt\utils\test_s3.py`

```

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/12/27
@Author  : mashenquan
@File    : test_s3.py
"""
# 导入所需的模块
import uuid
from pathlib import Path
import aiofiles
import mock
import pytest
# 导入自定义模块
from metagpt.config import CONFIG
from metagpt.utils.common import aread
from metagpt.utils.s3 import S3

# 使用 pytest 的异步测试装饰器
@pytest.mark.asyncio
# 使用 mock.patch 修饰测试函数，模拟 aioboto3.Session
@mock.patch("aioboto3.Session")
async def test_s3(mock_session_class):
    # 设置模拟响应
    data = await aread(__file__, "utf-8")
    mock_session_object = mock.Mock()
    reader_mock = mock.AsyncMock()
    reader_mock.read.side_effect = [data.encode("utf-8"), b"", data.encode("utf-8")]
    type(reader_mock).url = mock.PropertyMock(return_value="https://mock")
    mock_client = mock.AsyncMock()
    mock_client.put_object.return_value = None
    mock_client.get_object.return_value = {"Body": reader_mock}
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_session_object.client.return_value = mock_client
    mock_session_class.return_value = mock_session_object

    # 前提条件
    # assert CONFIG.S3_ACCESS_KEY and CONFIG.S3_ACCESS_KEY != "YOUR_S3_ACCESS_KEY"
    # assert CONFIG.S3_SECRET_KEY and CONFIG.S3_SECRET_KEY != "YOUR_S3_SECRET_KEY"
    # assert CONFIG.S3_ENDPOINT_URL and CONFIG.S3_ENDPOINT_URL != "YOUR_S3_ENDPOINT_URL"
    # assert CONFIG.S3_BUCKET and CONFIG.S3_BUCKET != "YOUR_S3_BUCKET"

    # 创建 S3 连接对象
    conn = S3()
    # 断言 S3 连接对象有效
    assert conn.is_valid
    # 设置对象名称
    object_name = "unittest.bak"
    # 上传文件到 S3
    await conn.upload_file(bucket=CONFIG.S3_BUCKET, local_path=__file__, object_name=object_name)
    # 创建本地文件路径
    pathname = (Path(__file__).parent / uuid.uuid4().hex).with_suffix(".bak")
    # 删除已存在的文件
    pathname.unlink(missing_ok=True)
    # 从 S3 下载文件到本地
    await conn.download_file(bucket=CONFIG.S3_BUCKET, object_name=object_name, local_path=str(pathname))
    # 断言文件存在
    assert pathname.exists()
    # 获取对象的 URL
    url = await conn.get_object_url(bucket=CONFIG.S3_BUCKET, object_name=object_name)
    # 断言 URL 存在
    assert url
    # 获取对象的二进制数据
    bin_data = await conn.get_object(bucket=CONFIG.S3_BUCKET, object_name=object_name)
    # 断言二进制数据存在
    assert bin_data
    # 使用 aiofiles 打开文件，读取数据
    async with aiofiles.open(__file__, mode="r", encoding="utf-8") as reader:
        data = await reader.read()
    # 缓存数据到 S3
    res = await conn.cache(data, ".bak", "script")
    # 断言结果包含 "http"
    assert "http" in res

    # 模拟会话环境
    type(reader_mock).url = mock.PropertyMock(return_value="")
    old_options = CONFIG.options.copy()
    new_options = old_options.copy()
    new_options["S3_ACCESS_KEY"] = "YOUR_S3_ACCESS_KEY"
    CONFIG.set_context(new_options)
    try:
        conn = S3()
        # 断言 S3 连接对象无效
        assert not conn.is_valid
        # 缓存数据到 S3，预期结果为假
        res = await conn.cache("ABC", ".bak", "script")
        assert not res
    finally:
        CONFIG.set_context(old_options)

    # 关闭文件读取器
    await reader.close()

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```