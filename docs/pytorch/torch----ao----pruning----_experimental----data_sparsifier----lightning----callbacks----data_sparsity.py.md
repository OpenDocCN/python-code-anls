# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\lightning\callbacks\data_sparsity.py`

```py
# mypy: allow-untyped-defs
# 从 collections 模块中导入 defaultdict 类
from collections import defaultdict
# 从 copy 模块中导入 deepcopy 函数
from copy import deepcopy
# 从 typing 模块中导入 Any, Optional, Dict, TYPE_CHECKING 类型
from typing import Any, Optional, Dict, TYPE_CHECKING
# 导入 pytorch_lightning 库，忽略类型检查错误
import pytorch_lightning as pl  # type: ignore[import]

# 从 _data_sparstity_utils 模块中导入以下函数
from ._data_sparstity_utils import (
    _attach_model_to_data_sparsifier,
    _log_sparsified_level,
    _get_valid_name
)

# 如果 TYPE_CHECKING 为真，导入 torch 模块
if TYPE_CHECKING:
    import torch

# 定义 PostTrainingDataSparsity 类，继承自 pl.callbacks.Callback
class PostTrainingDataSparsity(pl.callbacks.Callback):
    """Lightning callback that enables post-training sparsity.

    This callback aims to sparsify the model inside lightning module after training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_class (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this class is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            once the training completes.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

    Hooks implemented:
        on_fit_end()
            1. copies the model and attaches it to the sparsifier
            2. sparsier step() is called
            3. squashes the mask()
    """
    
    # 初始化方法，接收 data_sparsifier_class 和 data_sparsifier_args 两个参数
    def __init__(self, data_sparsifier_class, data_sparsifier_args):
        super().__init__()
        # 将参数保存为对象属性
        self.data_sparsifier_class = data_sparsifier_class
        self.data_sparsifier_args = data_sparsifier_args
        self.data_sparsifier: Any = None
        self.sparsified: Optional[torch.nn.Module] = None

    # on_fit_end 方法，在训练结束时调用
    def on_fit_end(self, trainer, pl_module) -> None:
        # 深度复制模型，并设置为评估模式
        self.sparsified = deepcopy(pl_module.model).eval()
        # 使用给定参数实例化数据稀疏化对象
        self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)

        # 将稀疏化后的模型与数据稀疏化对象关联
        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)

        # 执行数据稀疏化的步骤
        self.data_sparsifier.step()

        # 压缩稀疏化对象的掩码（mask）
        self.data_sparsifier.squash_mask()  # currently squashes params for all mask

        # 记录稀疏化水平
        _log_sparsified_level(self.sparsified, self.data_sparsifier)


# 定义 TrainingAwareDataSparsity 类，继承自 pl.callbacks.Callback
class TrainingAwareDataSparsity(pl.callbacks.Callback):
    """Lightning callback that enables in-training sparsity.

    This callback aims to sparsify the model inside lightning module during training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified
    """
    pass  # Placeholder for future implementation
    def __init__(self, data_sparsifier_class, data_sparsifier_args,
                 data_scheduler_class, data_scheduler_args):
        super().__init__()
        # data sparsifier objects
        self.data_sparsifier_class = data_sparsifier_class  # 存储数据稀疏化器类
        self.data_sparsifier_args = data_sparsifier_args    # 存储数据稀疏化器参数

        # scheduler objects
        self.data_scheduler_class = data_scheduler_class    # 存储数据调度器类
        self.data_scheduler_args = data_scheduler_args      # 存储数据调度器参数

        # fields
        self.data_sparsifier: Any = None                    # 初始化数据稀疏化器为空
        self.data_scheduler: Any = None                     # 初始化数据调度器为空
        self.sparsified: Optional[torch.nn.Module] = None   # 初始化稀疏化后的模型为可选的空

        self.data_sparsifier_state_dict: Any = None         # 初始化数据稀疏化器状态字典为空

    def on_train_start(self, trainer, pl_module) -> None:
        # create sparsifier
        self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)
        self.sparsified = deepcopy(pl_module.model)

        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)  # 将模型附加到数据稀疏化器上，以便在调度器中填充基本sl

        # create scheduler
        args = deepcopy(self.data_scheduler_args)
        args['data_sparsifier'] = self.data_sparsifier
        self.data_scheduler = self.data_scheduler_class(**args)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.data_sparsifier_state_dict is None:
            return  # 可能是第一个 epoch，无需加载状态字典

        # load the existing config for each data
        self.data_sparsifier.load_state_dict(self.data_sparsifier_state_dict)
    # 根据给定的 `pl_module` 创建基于状态的配置字典
    def __create_config_based_on_state(self, pl_module):
        # 使用 defaultdict 创建一个空的字典作为配置字典
        config: Dict = defaultdict()
        # 如果数据稀疏化状态字典为 None，则直接返回空的配置字典
        if self.data_sparsifier_state_dict is None:
            return config
        # 遍历模型中所有命名参数
        for name, _ in pl_module.model.named_parameters():
            # 获取有效的参数名称
            valid_name = _get_valid_name(name)
            # 将有效名称映射到数据稀疏化对象的数据组中，构建配置字典
            config[valid_name] = self.data_sparsifier.data_groups[valid_name]
        
        # 返回构建好的配置字典
        return config

    # 在训练每个 epoch 结束时调用的方法
    def on_train_epoch_end(self, trainer, pl_module):
        # 深度复制当前的模型并保存到 self.sparsified
        self.sparsified = deepcopy(pl_module.model)
        # 根据当前状态创建基于状态的配置字典
        config = self.__create_config_based_on_state(pl_module)

        # 将模型与数据稀疏化器关联，并传入配置字典作为参数
        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier, config=config)
        # 数据稀疏化器执行一步操作
        self.data_sparsifier.step()
        # 数据调度器执行一步操作
        self.data_scheduler.step()

        # 更新数据稀疏化器的状态字典为当前状态
        self.data_sparsifier_state_dict = self.data_sparsifier.state_dict()

    # 在训练结束时调用的方法
    def on_train_end(self, trainer, pl_module):
        # 压缩数据稀疏化器的掩码（即清空掩码）
        self.data_sparsifier.squash_mask()
```