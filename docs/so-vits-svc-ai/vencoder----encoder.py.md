# `so-vits-svc\vencoder\encoder.py`

```
# 定义一个名为SpeechEncoder的类
class SpeechEncoder(object):
    # 初始化方法，设置默认的vec_path和device参数
    def __init__(self, vec_path="pretrain/checkpoint_best_legacy_500.pt", device=None):
        # 初始化model属性为None
        self.model = None  # This is Model
        # 初始化hidden_dim属性为768
        self.hidden_dim = 768
        # pass表示占位符，暂时不执行任何操作

    # 定义一个名为encoder的方法，接受wav作为输入
    def encoder(self, wav):
        """
        input: wav:[signal_length]
        output: embedding:[batchsize,hidden_dim,wav_frame]
        """
        # pass表示占位符，暂时不执行任何操作
        pass
```