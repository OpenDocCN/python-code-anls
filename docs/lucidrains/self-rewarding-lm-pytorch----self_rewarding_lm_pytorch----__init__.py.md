# `.\lucidrains\self-rewarding-lm-pytorch\self_rewarding_lm_pytorch\__init__.py`

```py
# 导入自我奖励语言模型训练器和奖励配置
from self_rewarding_lm_pytorch.self_rewarding_lm_pytorch import (
    SelfRewardingTrainer,
    RewardConfig
)

# 导入 SPIN 模型和 SPIN 训练器
from self_rewarding_lm_pytorch.spin import (
    SPIN,
    SPINTrainer,
)

# 导入 DPO 模型和 DPO 训练器
from self_rewarding_lm_pytorch.dpo import (
    DPO,
    DPOTrainer,
)

# 导入创建模拟数据集的函数
from self_rewarding_lm_pytorch.mocks import create_mock_dataset

# 导入自我奖励语言模型微调配置
from self_rewarding_lm_pytorch.self_rewarding_lm_pytorch import (
    SFTConfig,
    SelfRewardDPOConfig,
    ExternalRewardDPOConfig,
    SelfPlayConfig,
    create_default_paper_config
)
```