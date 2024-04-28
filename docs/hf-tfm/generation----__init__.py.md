# `.\transformers\generation\__init__.py`

```
# 版权声明和许可信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ..utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_tf_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_utils": ["GenerationConfig"],
    "streamers": ["TextIteratorStreamer", "TextStreamer"],
}

# 尝试导入 torch 模块，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加以下模块到导入结构中
    _import_structure["beam_constraints"] = [
        "Constraint",
        "ConstraintListState",
        "DisjunctiveConstraint",
        "PhrasalConstraint",
    ]
    _import_structure["beam_search"] = [
        "BeamHypotheses",
        "BeamScorer",
        "BeamSearchScorer",
        "ConstrainedBeamSearchScorer",
    ]
    _import_structure["logits_process"] = [
        "AlternatingCodebooksLogitsProcessor",
        "ClassifierFreeGuidanceLogitsProcessor",
        "EncoderNoRepeatNGramLogitsProcessor",
        "EncoderRepetitionPenaltyLogitsProcessor",
        "EpsilonLogitsWarper",
        "EtaLogitsWarper",
        "ExponentialDecayLengthPenalty",
        "ForcedBOSTokenLogitsProcessor",
        "ForcedEOSTokenLogitsProcessor",
        "ForceTokensLogitsProcessor",
        "HammingDiversityLogitsProcessor",
        "InfNanRemoveLogitsProcessor",
        "LogitNormalization",
        "LogitsProcessor",
        "LogitsProcessorList",
        "LogitsWarper",
        "MinLengthLogitsProcessor",
        "MinNewTokensLengthLogitsProcessor",
        "NoBadWordsLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "SequenceBiasLogitsProcessor",
        "SuppressTokensLogitsProcessor",
        "SuppressTokensAtBeginLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
        "TypicalLogitsWarper",
        "UnbatchedClassifierFreeGuidanceLogitsProcessor",
        "WhisperTimeStampLogitsProcessor",
    ]
    _import_structure["stopping_criteria"] = [
        "MaxNewTokensCriteria",
        "MaxLengthCriteria",
        "MaxTimeCriteria",
        "StoppingCriteria",
        "StoppingCriteriaList",
        "validate_stopping_criteria",
    ]
    # 将一组工具函数的结构信息添加到_import_structure字典中，该字典的键是模块名，值是包含在该模块中的函数列表
    _import_structure["utils"] = [
        # 将 GenerationMixin 类添加到 utils 模块中
        "GenerationMixin",
        # 将 top_k_top_p_filtering 函数添加到 utils 模块中
        "top_k_top_p_filtering",
        # 将 GreedySearchEncoderDecoderOutput 类添加到 utils 模块中
        "GreedySearchEncoderDecoderOutput",
        # 将 GreedySearchDecoderOnlyOutput 类添加到 utils 模块中
        "GreedySearchDecoderOnlyOutput",
        # 将 SampleEncoderDecoderOutput 类添加到 utils 模块中
        "SampleEncoderDecoderOutput",
        # 将 SampleDecoderOnlyOutput 类添加到 utils 模块中
        "SampleDecoderOnlyOutput",
        # 将 BeamSearchEncoderDecoderOutput 类添加到 utils 模块中
        "BeamSearchEncoderDecoderOutput",
        # 将 BeamSearchDecoderOnlyOutput 类添加到 utils 模块中
        "BeamSearchDecoderOnlyOutput",
        # 将 BeamSampleEncoderDecoderOutput 类添加到 utils 模块中
        "BeamSampleEncoderDecoderOutput",
        # 将 BeamSampleDecoderOnlyOutput 类添加到 utils 模块中
        "BeamSampleDecoderOnlyOutput",
        # 将 ContrastiveSearchEncoderDecoderOutput 类添加到 utils 模块中
        "ContrastiveSearchEncoderDecoderOutput",
        # 将 ContrastiveSearchDecoderOnlyOutput 类添加到 utils 模块中
        "ContrastiveSearchDecoderOnlyOutput",
        # 将 GenerateBeamDecoderOnlyOutput 类添加到 utils 模块中
        "GenerateBeamDecoderOnlyOutput",
        # 将 GenerateBeamEncoderDecoderOutput 类添加到 utils 模块中
        "GenerateBeamEncoderDecoderOutput",
        # 将 GenerateDecoderOnlyOutput 类添加到 utils 模块中
        "GenerateDecoderOnlyOutput",
        # 将 GenerateEncoderDecoderOutput 类添加到 utils 模块中
        "GenerateEncoderDecoderOutput",
    ]
# 尝试导入 TensorFlow 库，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将 TensorFlow logits 处理相关模块添加到导入结构中
    _import_structure["tf_logits_process"] = [
        "TFForcedBOSTokenLogitsProcessor",
        "TFForcedEOSTokenLogitsProcessor",
        ...
    ]
    # 将 TensorFlow 工具相关模块添加到导入结构中
    _import_structure["tf_utils"] = [
        "TFGenerationMixin",
        "tf_top_k_top_p_filtering",
        ...
    ]

# 尝试导入 Flax 库，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将 Flax logits 处理相关模块添加到导入结构中
    _import_structure["flax_logits_process"] = [
        "FlaxForcedBOSTokenLogitsProcessor",
        "FlaxForcedEOSTokenLogitsProcessor",
        ...
    ]
    # 将 Flax 工具相关模块添加到导入结构中
    _import_structure["flax_utils"] = [
        "FlaxGenerationMixin",
        "FlaxGreedySearchOutput",
        ...
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入生成配置相关模块
    from .configuration_utils import GenerationConfig
    # 导入文本迭代器流和文本流相关模块
    from .streamers import TextIteratorStreamer, TextStreamer

    # 尝试导入 PyTorch 库，如果不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 导入所需模块，这些模块在不同情况下会用于各种约束、搜索、处理、停止标准和实用工具
    else:
        from .beam_constraints import Constraint, ConstraintListState, DisjunctiveConstraint, PhrasalConstraint
        from .beam_search import BeamHypotheses, BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
        from .logits_process import (
            AlternatingCodebooksLogitsProcessor,  # 处理概率分布的后处理器，用于音频生成任务
            ClassifierFreeGuidanceLogitsProcessor,  # 自由导向的分类器后处理器
            EncoderNoRepeatNGramLogitsProcessor,  # 编码器不重复 N-Gram 后处理器
            EncoderRepetitionPenaltyLogitsProcessor,  # 编码器重复惩罚后处理器
            EpsilonLogitsWarper,  # Epsilon 概率变换器
            EtaLogitsWarper,  # Eta 概率变换器
            ExponentialDecayLengthPenalty,  # 指数衰减长度惩罚
            ForcedBOSTokenLogitsProcessor,  # 强制开始符号的后处理器
            ForcedEOSTokenLogitsProcessor,  # 强制结束符号的后处理器
            ForceTokensLogitsProcessor,  # 强制特定 token 的后处理器
            HammingDiversityLogitsProcessor,  # Hamming 多样性后处理器
            InfNanRemoveLogitsProcessor,  # 移除无穷大或 NaN 的后处理器
            LogitNormalization,  # 对数归一化处理器
            LogitsProcessor,  # 概率分布后处理器的基类
            LogitsProcessorList,  # 概率分布后处理器列表
            LogitsWarper,  # 概率分布变换器的基类
            MinLengthLogitsProcessor,  # 最小长度后处理器
            MinNewTokensLengthLogitsProcessor,  # 最小新 token 长度后处理器
            NoBadWordsLogitsProcessor,  # 无不良词语后处理器
            NoRepeatNGramLogitsProcessor,  # 无重复 N-Gram 后处理器
            PrefixConstrainedLogitsProcessor,  # 前缀约束的概率分布后处理器
            RepetitionPenaltyLogitsProcessor,  # 重复惩罚概率分布后处理器
            SequenceBiasLogitsProcessor,  # 序列偏置概率分布后处理器
            SuppressTokensAtBeginLogitsProcessor,  # 抑制开始 token 的概率分布后处理器
            SuppressTokensLogitsProcessor,  # 抑制特定 token 的概率分布后处理器
            TemperatureLogitsWarper,  # 温度概率变换器
            TopKLogitsWarper,  # Top-K 概率变换器
            TopPLogitsWarper,  # Top-P 概率变换器
            TypicalLogitsWarper,  # 典型概率变换器
            UnbatchedClassifierFreeGuidanceLogitsProcessor,  # 非批量自由导向的分类器后处理器
            WhisperTimeStampLogitsProcessor,  # Whisper 时间戳后处理器
        )
        from .stopping_criteria import (
            MaxLengthCriteria,  # 最大长度停止标准
            MaxNewTokensCriteria,  # 最大新 token 数停止标准
            MaxTimeCriteria,  # 最大时间停止标准
            StoppingCriteria,  # 停止标准的基类
            StoppingCriteriaList,  # 停止标准列表
            validate_stopping_criteria,  # 验证停止标准的函数
        )
        from .utils import (
            BeamSampleDecoderOnlyOutput,  # Beam 搜索采样解码器的输出
            BeamSampleEncoderDecoderOutput,  # Beam 搜索采样编码器-解码器的输出
            BeamSearchDecoderOnlyOutput,  # Beam 搜索解码器的输出
            BeamSearchEncoderDecoderOutput,  # Beam 搜索编码器-解码器的输出
            ContrastiveSearchDecoderOnlyOutput,  # 对比搜索解码器的输出
            ContrastiveSearchEncoderDecoderOutput,  # 对比搜索编码器-解码器的输出
            GenerateBeamDecoderOnlyOutput,  # 生成 Beam 解码器的输出
            GenerateBeamEncoderDecoderOutput,  # 生成 Beam 编码器-解码器的输出
            GenerateDecoderOnlyOutput,  # 生成解码器的输出
            GenerateEncoderDecoderOutput,  # 生成编码器-解码器的输出
            GenerationMixin,  # 生成混合类
            GreedySearchDecoderOnlyOutput,  # 贪婪搜索解码器的输出
            GreedySearchEncoderDecoderOutput,  # 贪婪搜索编码器-解码器的输出
            SampleDecoderOnlyOutput,  # 采样解码器的输出
            SampleEncoderDecoderOutput,  # 采样编码器-解码器的输出
            top_k_top_p_filtering,  # Top-K Top-P 过滤器
        )

    # 检查是否可用 TensorFlow，如果不可用，则引发自定义的异常 OptionalDependencyNotAvailable
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 不可用，则忽略该异常
        pass
    # 如果不是第一种情况，即使用 TensorFlow 的情况，则导入 TensorFlow 相关模块和类
    else:
        # 从当前目录中导入 TensorFlow 下的 logits 处理相关模块和类
        from .tf_logits_process import (
            TFForcedBOSTokenLogitsProcessor,  # 强制开头标记 logits 处理器
            TFForcedEOSTokenLogitsProcessor,  # 强制结尾标记 logits 处理器
            TFForceTokensLogitsProcessor,      # 强制 token logits 处理器
            TFLogitsProcessor,                 # logits 处理器基类
            TFLogitsProcessorList,             # logits 处理器列表
            TFLogitsWarper,                    # logits 处理器包装器
            TFMinLengthLogitsProcessor,        # 最小长度 logits 处理器
            TFNoBadWordsLogitsProcessor,       # 无不良词语 logits 处理器
            TFNoRepeatNGramLogitsProcessor,    # 无重复 N 元组 logits 处理器
            TFRepetitionPenaltyLogitsProcessor,# 重复惩罚 logits 处理器
            TFSuppressTokensAtBeginLogitsProcessor,  # 开头抑制 token logits 处理器
            TFSuppressTokensLogitsProcessor,   # 抑制 token logits 处理器
            TFTemperatureLogitsWarper,         # 温度 logits 包装器
            TFTopKLogitsWarper,                # 前 K 个 logits 包装器
            TFTopPLogitsWarper,                # 前 P 个 logits 包装器
        )
        # 从当前目录中导入 TensorFlow 下的工具模块和类
        from .tf_utils import (
            TFBeamSampleDecoderOnlyOutput,     # Beam Sample 解码器输出
            TFBeamSampleEncoderDecoderOutput,  # Beam Sample 编码器解码器输出
            TFBeamSearchDecoderOnlyOutput,     # Beam Search 解码器输出
            TFBeamSearchEncoderDecoderOutput,  # Beam Search 编码器解码器输出
            TFContrastiveSearchDecoderOnlyOutput,  # 对比搜索解码器输出
            TFContrastiveSearchEncoderDecoderOutput,  # 对比搜索编码器解码器输出
            TFGenerationMixin,                 # 生成 mixin
            TFGreedySearchDecoderOnlyOutput,   # 贪婪搜索解码器输出
            TFGreedySearchEncoderDecoderOutput,# 贪婪搜索编码器解码器输出
            TFSampleDecoderOnlyOutput,         # Sample 解码器输出
            TFSampleEncoderDecoderOutput,      # Sample 编码器解码器输出
            tf_top_k_top_p_filtering,          # Top-K Top-P 过滤函数
        )

    # 尝试检查是否可用 Flax，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果 Flax 可用，则导入 Flax 相关模块和类
    else:
        # 从当前目录中导入 Flax 下的 logits 处理相关模块和类
        from .flax_logits_process import (
            FlaxForcedBOSTokenLogitsProcessor,  # 强制开头标记 logits 处理器
            FlaxForcedEOSTokenLogitsProcessor,  # 强制结尾标记 logits 处理器
            FlaxForceTokensLogitsProcessor,     # 强制 token logits 处理器
            FlaxLogitsProcessor,                # logits 处理器基类
            FlaxLogitsProcessorList,            # logits 处理器列表
            FlaxLogitsWarper,                   # logits 处理器包装器
            FlaxMinLengthLogitsProcessor,       # 最小长度 logits 处理器
            FlaxSuppressTokensAtBeginLogitsProcessor,  # 开头抑制 token logits 处理器
            FlaxSuppressTokensLogitsProcessor,  # 抑制 token logits 处理器
            FlaxTemperatureLogitsWarper,        # 温度 logits 包装器
            FlaxTopKLogitsWarper,               # 前 K 个 logits 包装器
            FlaxTopPLogitsWarper,               # 前 P 个 logits 包装器
            FlaxWhisperTimeStampLogitsProcessor,# 密语时间戳 logits 处理器
        )
        # 从当前目录中导入 Flax 下的工具模块和类
        from .flax_utils import (
            FlaxBeamSearchOutput,               # Beam Search 输出
            FlaxGenerationMixin,                # 生成 mixin
            FlaxGreedySearchOutput,             # 贪婪搜索输出
            FlaxSampleOutput,                   # Sample 输出
        )
```  
# 如果不在主模块中，则导入sys模块
import sys
# 将当前模块添加到sys.modules字典中，使用_LazyModule延迟加载模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```