# `MetaGPT\tests\metagpt\tools\test_azure_tts.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/1 22:50
@Author  : alexanderwu
@File    : test_azure_tts.py
@Modified By: mashenquan, 2023-8-9, add more text formatting options
@Modified By: mashenquan, 2023-8-17, move to `tools` folder.
"""

# 导入所需的模块
import pytest
from azure.cognitiveservices.speech import ResultReason

# 从metagpt.config中导入CONFIG对象
from metagpt.config import CONFIG
# 从metagpt.tools.azure_tts中导入AzureTTS类
from metagpt.tools.azure_tts import AzureTTS

# 使用pytest.mark.asyncio装饰器标记异步测试函数
@pytest.mark.asyncio
async def test_azure_tts():
    # 先决条件
    assert CONFIG.AZURE_TTS_SUBSCRIPTION_KEY and CONFIG.AZURE_TTS_SUBSCRIPTION_KEY != "YOUR_API_KEY"
    assert CONFIG.AZURE_TTS_REGION

    # 创建AzureTTS对象
    azure_tts = AzureTTS(subscription_key="", region="")
    # 设置待合成的文本
    text = """
        女儿看见父亲走了进来，问道：
            <mstts:express-as role="YoungAdultFemale" style="calm">
                “您来的挺快的，怎么过来的？”
            </mstts:express-as>
            父亲放下手提包，说：
            <mstts:express-as role="OlderAdultMale" style="calm">
                “Writing a binary file in Python is similar to writing a regular text file, but you'll work with bytes instead of strings.”
            </mstts:express-as>
        """
    # 设置输出路径
    path = CONFIG.workspace_path / "tts"
    path.mkdir(exist_ok=True, parents=True)
    filename = path / "girl.wav"
    filename.unlink(missing_ok=True)
    # 调用AzureTTS对象的synthesize_speech方法进行语音合成
    result = await azure_tts.synthesize_speech(
        lang="zh-CN", voice="zh-CN-XiaomoNeural", text=text, output_file=str(filename)
    )
    # 打印合成结果
    print(result)
    # 断言合成结果
    assert result
    assert result.audio_data
    assert result.reason == ResultReason.SynthesizingAudioCompleted
    assert filename.exists()

# 如果当前文件被直接执行，则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```