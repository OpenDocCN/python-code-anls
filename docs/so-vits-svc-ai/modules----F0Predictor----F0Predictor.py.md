# `so-vits-svc\modules\F0Predictor\F0Predictor.py`

```
# 定义 F0Predictor 类，用于预测音频信号的基频
class F0Predictor(object):
    # 计算音频信号的基频
    def compute_f0(self, wav, p_len):
        '''
        input: wav:[signal_length]  # 输入参数为音频信号，长度为 signal_length
               p_len:int  # 输入参数为整数，表示某种长度
        output: f0:[signal_length//hop_length]  # 输出为基频，长度为 signal_length//hop_length
        '''
        pass  # 占位符，表示该方法暂时不实现

    # 计算音频信号的基频和声门开关
    def compute_f0_uv(self, wav, p_len):
        '''
        input: wav:[signal_length]  # 输入参数为音频信号，长度为 signal_length
               p_len:int  # 输入参数为整数，表示某种长度
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]  # 输出为基频和声门开关，长度为 signal_length//hop_length
        '''
        pass  # 占位符，表示该方法暂时不实现
```