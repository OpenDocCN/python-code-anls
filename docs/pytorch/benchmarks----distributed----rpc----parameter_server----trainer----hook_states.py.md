# `.\pytorch\benchmarks\distributed\rpc\parameter_server\trainer\hook_states.py`

```py
class BasicHookState:
    # 初始化方法，创建一个包含通信钩子在训练过程中所需状态信息的类
    def __init__(self, cref, process_group):
        """
        A class that holds state information that is needed by the communication hook
        during the training algorithm.
        Args:
            cref (DdpTrainer): reference to the self keyword of the trainer instance
            process_group (ProcessGroup): distributed process group
        """
        self.cref = cref
        self.process_group = process_group
        self.batch_number = -1  # 初始化批次号为-1

    # 返回编码的键，表示当前批次和桶索引的方法
    def get_key(self, bucket_index):
        """
        A method that returns an encoded key that represents the current batch and
        bucket index.
        Args:
            bucket_index (int): index of the bucket being processed in backward
        """
        return f"{self.batch_number},{bucket_index}"

    # 增加批次号的方法
    def next_batch(self):
        """
        A method that increments batch_number by 1.
        """
        self.batch_number += 1
```