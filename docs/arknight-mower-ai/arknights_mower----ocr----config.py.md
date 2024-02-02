# `arknights-mower\arknights_mower\ocr\config.py`

```py
# 从上级目录中导入__rootdir__变量
from .. import __rootdir__
# 设置dbnet模型路径为__rootdir__/models/dbnet.onnx
dbnet_model_path = f'{__rootdir__}/models/dbnet.onnx'
# 设置crnn模型路径为__rootdir__/models/crnn_lite_lstm.onnx
crnn_model_path = f'{__rootdir__}/models/crnn_lite_lstm.onnx'
```