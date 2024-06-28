# `.\generation\__init__.py`

```
# 引入类型检查模块的条件语句，用于确定当前环境是否支持类型检查
from typing import TYPE_CHECKING

# 引入必要的依赖和模块，用于检查和延迟加载
from ..utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_tf_available, is_torch_available

# 定义需要导入的模块结构字典
_import_structure = {
    "configuration_utils": ["GenerationConfig", "GenerationMode"],  # 配置工具模块
    "streamers": ["TextIteratorStreamer", "TextStreamer"],  # 数据流处理模块
}

# 尝试导入 torch 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加以下模块到导入结构字典
    _import_structure["beam_constraints"] = [
        "Constraint",  # 约束条件模块
        "ConstraintListState",  # 约束条件列表状态模块
        "DisjunctiveConstraint",  # 分离约束模块
        "PhrasalConstraint",  # 短语约束模块
    ]
    _import_structure["beam_search"] = [
        "BeamHypotheses",  # 搜索假设模块
        "BeamScorer",  # 搜索评分器模块
        "BeamSearchScorer",  # 搜索评分器模块
        "ConstrainedBeamSearchScorer",  # 约束搜索评分器模块
    ]
    _import_structure["candidate_generator"] = [
        "AssistedCandidateGenerator",  # 候选生成辅助模块
        "CandidateGenerator",  # 候选生成器模块
        "PromptLookupCandidateGenerator",  # 提示查找候选生成器模块
    ]
    _import_structure["logits_process"] = [
        "AlternatingCodebooksLogitsProcessor",  # 替换码本逻辑处理器模块
        "ClassifierFreeGuidanceLogitsProcessor",  # 免分类器引导逻辑处理器模块
        "EncoderNoRepeatNGramLogitsProcessor",  # 编码器无重复 n-gram 逻辑处理器模块
        "EncoderRepetitionPenaltyLogitsProcessor",  # 编码器重复惩罚逻辑处理器模块
        "EpsilonLogitsWarper",  # Epsilon 逻辑扭曲器模块
        "EtaLogitsWarper",  # Eta 逻辑扭曲器模块
        "ExponentialDecayLengthPenalty",  # 指数衰减长度惩罚模块
        "ForcedBOSTokenLogitsProcessor",  # 强制 BOS 标记逻辑处理器模块
        "ForcedEOSTokenLogitsProcessor",  # 强制 EOS 标记逻辑处理器模块
        "ForceTokensLogitsProcessor",  # 强制令牌逻辑处理器模块
        "HammingDiversityLogitsProcessor",  # 汉明多样性逻辑处理器模块
        "InfNanRemoveLogitsProcessor",  # 无穷大和无效值移除逻辑处理器模块
        "LogitNormalization",  # Logit 归一化模块
        "LogitsProcessor",  # Logits 处理器模块
        "LogitsProcessorList",  # Logits 处理器列表模块
        "LogitsWarper",  # Logits 扭曲器模块
        "MinLengthLogitsProcessor",  # 最小长度逻辑处理器模块
        "MinNewTokensLengthLogitsProcessor",  # 最小新令牌长度逻辑处理器模块
        "NoBadWordsLogitsProcessor",  # 无不良词语逻辑处理器模块
        "NoRepeatNGramLogitsProcessor",  # 无重复 n-gram 逻辑处理器模块
        "PrefixConstrainedLogitsProcessor",  # 前缀约束逻辑处理器模块
        "RepetitionPenaltyLogitsProcessor",  # 重复惩罚逻辑处理器模块
        "SequenceBiasLogitsProcessor",  # 序列偏置逻辑处理器模块
        "SuppressTokensLogitsProcessor",  # 抑制令牌逻辑处理器模块
        "SuppressTokensAtBeginLogitsProcessor",  # 在开头抑制令牌逻辑处理器模块
        "TemperatureLogitsWarper",  # 温度逻辑扭曲器模块
        "TopKLogitsWarper",  # Top-K 逻辑扭曲器模块
        "TopPLogitsWarper",  # Top-P 逻辑扭曲器模块
        "TypicalLogitsWarper",  # 典型逻辑扭曲器模块
        "UnbatchedClassifierFreeGuidanceLogitsProcessor",  # 未分批免分类器引导逻辑处理器模块
        "WhisperTimeStampLogitsProcessor",  # Whisper 时间戳逻辑处理器模块
    ]
    # 将停止条件模块的类名列表添加到_import_structure字典中的"stopping_criteria"键下
    _import_structure["stopping_criteria"] = [
        "MaxNewTokensCriteria",  # 最大新标记数条件
        "MaxLengthCriteria",  # 最大长度条件
        "MaxTimeCriteria",  # 最大时间条件
        "StoppingCriteria",  # 停止条件基类
        "StoppingCriteriaList",  # 停止条件列表
        "validate_stopping_criteria",  # 验证停止条件函数
    ]
    
    # 将实用工具模块的类名列表添加到_import_structure字典中的"utils"键下
    _import_structure["utils"] = [
        "GenerationMixin",  # 生成混合类
        "GreedySearchEncoderDecoderOutput",  # 贪婪搜索编码器解码器输出
        "GreedySearchDecoderOnlyOutput",  # 贪婪搜索仅解码器输出
        "SampleEncoderDecoderOutput",  # 样本编码器解码器输出
        "SampleDecoderOnlyOutput",  # 样本仅解码器输出
        "BeamSearchEncoderDecoderOutput",  # Beam搜索编码器解码器输出
        "BeamSearchDecoderOnlyOutput",  # Beam搜索仅解码器输出
        "BeamSampleEncoderDecoderOutput",  # Beam样本编码器解码器输出
        "BeamSampleDecoderOnlyOutput",  # Beam样本仅解码器输出
        "ContrastiveSearchEncoderDecoderOutput",  # 对比搜索编码器解码器输出
        "ContrastiveSearchDecoderOnlyOutput",  # 对比搜索仅解码器输出
        "GenerateBeamDecoderOnlyOutput",  # 生成Beam解码器输出
        "GenerateBeamEncoderDecoderOutput",  # 生成Beam编码器解码器输出
        "GenerateDecoderOnlyOutput",  # 生成仅解码器输出
        "GenerateEncoderDecoderOutput",  # 生成编码器解码器输出
    ]
# 尝试检查是否可以使用 TensorFlow 库
try:
    # 如果 TensorFlow 不可用，引发 OptionalDependencyNotAvailable 异常
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果引发了 OptionalDependencyNotAvailable 异常，什么都不做，继续执行下一个代码块
    pass
else:
    # 如果没有引发异常，则将以下 TensorFlow 相关类添加到 _import_structure 字典中
    _import_structure["tf_logits_process"] = [
        "TFForcedBOSTokenLogitsProcessor",
        "TFForcedEOSTokenLogitsProcessor",
        "TFForceTokensLogitsProcessor",
        "TFLogitsProcessor",
        "TFLogitsProcessorList",
        "TFLogitsWarper",
        "TFMinLengthLogitsProcessor",
        "TFNoBadWordsLogitsProcessor",
        "TFNoRepeatNGramLogitsProcessor",
        "TFRepetitionPenaltyLogitsProcessor",
        "TFSuppressTokensAtBeginLogitsProcessor",
        "TFSuppressTokensLogitsProcessor",
        "TFTemperatureLogitsWarper",
        "TFTopKLogitsWarper",
        "TFTopPLogitsWarper",
    ]
    _import_structure["tf_utils"] = [
        "TFGenerationMixin",
        "TFGreedySearchDecoderOnlyOutput",
        "TFGreedySearchEncoderDecoderOutput",
        "TFSampleEncoderDecoderOutput",
        "TFSampleDecoderOnlyOutput",
        "TFBeamSearchEncoderDecoderOutput",
        "TFBeamSearchDecoderOnlyOutput",
        "TFBeamSampleEncoderDecoderOutput",
        "TFBeamSampleDecoderOnlyOutput",
        "TFContrastiveSearchEncoderDecoderOutput",
        "TFContrastiveSearchDecoderOnlyOutput",
    ]

# 尝试检查是否可以使用 Flax 库
try:
    # 如果 Flax 不可用，引发 OptionalDependencyNotAvailable 异常
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果引发了 OptionalDependencyNotAvailable 异常，什么都不做，继续执行下一个代码块
    pass
else:
    # 如果没有引发异常，则将以下 Flax 相关类添加到 _import_structure 字典中
    _import_structure["flax_logits_process"] = [
        "FlaxForcedBOSTokenLogitsProcessor",
        "FlaxForcedEOSTokenLogitsProcessor",
        "FlaxForceTokensLogitsProcessor",
        "FlaxLogitsProcessor",
        "FlaxLogitsProcessorList",
        "FlaxLogitsWarper",
        "FlaxMinLengthLogitsProcessor",
        "FlaxSuppressTokensAtBeginLogitsProcessor",
        "FlaxSuppressTokensLogitsProcessor",
        "FlaxTemperatureLogitsWarper",
        "FlaxTopKLogitsWarper",
        "FlaxTopPLogitsWarper",
        "FlaxWhisperTimeStampLogitsProcessor",
    ]
    _import_structure["flax_utils"] = [
        "FlaxGenerationMixin",
        "FlaxGreedySearchOutput",
        "FlaxSampleOutput",
        "FlaxBeamSearchOutput",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从相关模块导入特定类和函数
    from .configuration_utils import GenerationConfig, GenerationMode
    from .streamers import TextIteratorStreamer, TextStreamer
    
    try:
        # 如果 Torch 不可用，引发 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果引发了 OptionalDependencyNotAvailable 异常，什么都不做
        pass
    # 否则，从本地的beam_constraints模块中导入多个约束类和对象
    from .beam_constraints import Constraint, ConstraintListState, DisjunctiveConstraint, PhrasalConstraint
    # 从本地的beam_search模块中导入多个与beam搜索相关的类和对象
    from .beam_search import BeamHypotheses, BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
    # 从本地的candidate_generator模块中导入多个候选生成器类和对象
    from .candidate_generator import AssistedCandidateGenerator, CandidateGenerator, PromptLookupCandidateGenerator
    # 从本地的logits_process模块中导入多个logits处理类和对象
    from .logits_process import (
        AlternatingCodebooksLogitsProcessor,               # 处理交替码簿的logits处理器
        ClassifierFreeGuidanceLogitsProcessor,            # 无分类器指导的logits处理器
        EncoderNoRepeatNGramLogitsProcessor,              # 编码器不重复n-gram的logits处理器
        EncoderRepetitionPenaltyLogitsProcessor,         # 编码器重复惩罚的logits处理器
        EpsilonLogitsWarper,                             # Epsilon的logits调节器
        EtaLogitsWarper,                                 # Eta的logits调节器
        ExponentialDecayLengthPenalty,                    # 指数衰减长度惩罚
        ForcedBOSTokenLogitsProcessor,                   # 强制BOS标记的logits处理器
        ForcedEOSTokenLogitsProcessor,                   # 强制EOS标记的logits处理器
        ForceTokensLogitsProcessor,                      # 强制token的logits处理器
        HammingDiversityLogitsProcessor,                 # Hamming多样性的logits处理器
        InfNanRemoveLogitsProcessor,                     # 移除Inf和NaN的logits处理器
        LogitNormalization,                              # logits归一化处理器
        LogitsProcessor,                                 # logits处理器基类
        LogitsProcessorList,                             # logits处理器列表
        LogitsWarper,                                    # logits调节器基类
        MinLengthLogitsProcessor,                        # 最小长度的logits处理器
        MinNewTokensLengthLogitsProcessor,               # 最小新token长度的logits处理器
        NoBadWordsLogitsProcessor,                       # 无不良词语的logits处理器
        NoRepeatNGramLogitsProcessor,                    # 不重复n-gram的logits处理器
        PrefixConstrainedLogitsProcessor,                # 前缀约束的logits处理器
        RepetitionPenaltyLogitsProcessor,                # 重复惩罚的logits处理器
        SequenceBiasLogitsProcessor,                     # 序列偏置的logits处理器
        SuppressTokensAtBeginLogitsProcessor,            # 在开头抑制token的logits处理器
        SuppressTokensLogitsProcessor,                   # 抑制token的logits处理器
        TemperatureLogitsWarper,                         # 温度的logits调节器
        TopKLogitsWarper,                                # 前K个logits调节器
        TopPLogitsWarper,                                # Top-P的logits调节器
        TypicalLogitsWarper,                             # 典型的logits调节器
        UnbatchedClassifierFreeGuidanceLogitsProcessor,  # 无批处理分类器指导的logits处理器
        WhisperTimeStampLogitsProcessor,                 # Whisper时间戳的logits处理器
    )
    # 从本地的stopping_criteria模块中导入多个停止标准类和函数
    from .stopping_criteria import (
        MaxLengthCriteria,          # 最大长度标准
        MaxNewTokensCriteria,       # 最大新token标准
        MaxTimeCriteria,            # 最大时间标准
        StoppingCriteria,           # 停止标准基类
        StoppingCriteriaList,       # 停止标准列表
        validate_stopping_criteria, # 验证停止标准的函数
    )
    # 从本地的utils模块中导入多个实用类和对象，用于不同类型的解码和编码输出
    from .utils import (
        BeamSampleDecoderOnlyOutput,                    # Beam采样仅解码器输出
        BeamSampleEncoderDecoderOutput,                 # Beam采样编码器-解码器输出
        BeamSearchDecoderOnlyOutput,                    # Beam搜索仅解码器输出
        BeamSearchEncoderDecoderOutput,                 # Beam搜索编码器-解码器输出
        ContrastiveSearchDecoderOnlyOutput,             # 对比搜索仅解码器输出
        ContrastiveSearchEncoderDecoderOutput,          # 对比搜索编码器-解码器输出
        GenerateBeamDecoderOnlyOutput,                  # 生成Beam仅解码器输出
        GenerateBeamEncoderDecoderOutput,               # 生成Beam编码器-解码器输出
        GenerateDecoderOnlyOutput,                      # 生成仅解码器输出
        GenerateEncoderDecoderOutput,                   # 生成编码器-解码器输出
        GenerationMixin,                               # 生成Mixin
        GreedySearchDecoderOnlyOutput,                  # 贪婪搜索仅解码器输出
        GreedySearchEncoderDecoderOutput,               # 贪婪搜索编码器-解码器输出
        SampleDecoderOnlyOutput,                        # 采样仅解码器输出
        SampleEncoderDecoderOutput,                     # 采样编码器-解码器输出
    )

    # 尝试检查是否存在TensorFlow依赖，若不存在则引发OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():  # 如果TensorFlow不可用
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:  # 捕获OptionalDependencyNotAvailable异常
        pass  # 什么都不做，继续执行后续代码
    else:
        # 导入针对 TensorFlow 的 logits 处理模块
        from .tf_logits_process import (
            TFForcedBOSTokenLogitsProcessor,         # 强制开头标记的 logits 处理器
            TFForcedEOSTokenLogitsProcessor,         # 强制结尾标记的 logits 处理器
            TFForceTokensLogitsProcessor,            # 强制标记的 logits 处理器
            TFLogitsProcessor,                       # logits 处理器基类
            TFLogitsProcessorList,                   # logits 处理器列表
            TFLogitsWarper,                          # logits 调整器
            TFMinLengthLogitsProcessor,              # 最小长度的 logits 处理器
            TFNoBadWordsLogitsProcessor,             # 无不良词语的 logits 处理器
            TFNoRepeatNGramLogitsProcessor,          # 无重复 n-gram 的 logits 处理器
            TFRepetitionPenaltyLogitsProcessor,      # 重复惩罚的 logits 处理器
            TFSuppressTokensAtBeginLogitsProcessor,  # 开头抑制标记的 logits 处理器
            TFSuppressTokensLogitsProcessor,         # 抑制标记的 logits 处理器
            TFTemperatureLogitsWarper,               # 温度调整器
            TFTopKLogitsWarper,                      # 基于 top-k 的 logits 调整器
            TFTopPLogitsWarper,                      # 基于 top-p 的 logits 调整器
        )
        # 导入针对 TensorFlow 的实用工具模块
        from .tf_utils import (
            TFBeamSampleDecoderOnlyOutput,           # 仅解码器的 Beam Sample 输出
            TFBeamSampleEncoderDecoderOutput,        # 编码器解码器的 Beam Sample 输出
            TFBeamSearchDecoderOnlyOutput,           # 仅解码器的 Beam Search 输出
            TFBeamSearchEncoderDecoderOutput,        # 编码器解码器的 Beam Search 输出
            TFContrastiveSearchDecoderOnlyOutput,    # 仅解码器的对比搜索输出
            TFContrastiveSearchEncoderDecoderOutput, # 编码器解码器的对比搜索输出
            TFGenerationMixin,                       # 生成混合类
            TFGreedySearchDecoderOnlyOutput,         # 仅解码器的 Greedy Search 输出
            TFGreedySearchEncoderDecoderOutput,      # 编码器解码器的 Greedy Search 输出
            TFSampleDecoderOnlyOutput,               # 仅解码器的 Sample 输出
            TFSampleEncoderDecoderOutput,            # 编码器解码器的 Sample 输出
        )

    try:
        # 检查 Flax 是否可用，如果不可用则抛出异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 Flax 不可用，忽略该异常
        pass
    else:
        # 导入针对 Flax 的 logits 处理模块
        from .flax_logits_process import (
            FlaxForcedBOSTokenLogitsProcessor,           # 强制开头标记的 logits 处理器
            FlaxForcedEOSTokenLogitsProcessor,           # 强制结尾标记的 logits 处理器
            FlaxForceTokensLogitsProcessor,              # 强制标记的 logits 处理器
            FlaxLogitsProcessor,                         # logits 处理器基类
            FlaxLogitsProcessorList,                     # logits 处理器列表
            FlaxLogitsWarper,                            # logits 调整器
            FlaxMinLengthLogitsProcessor,                # 最小长度的 logits 处理器
            FlaxSuppressTokensAtBeginLogitsProcessor,    # 开头抑制标记的 logits 处理器
            FlaxSuppressTokensLogitsProcessor,           # 抑制标记的 logits 处理器
            FlaxTemperatureLogitsWarper,                 # 温度调整器
            FlaxTopKLogitsWarper,                        # 基于 top-k 的 logits 调整器
            FlaxTopPLogitsWarper,                        # 基于 top-p 的 logits 调整器
            FlaxWhisperTimeStampLogitsProcessor,         # Whisper 时间戳的 logits 处理器
        )
        # 导入针对 Flax 的实用工具模块
        from .flax_utils import (
            FlaxBeamSearchOutput,                       # Flax Beam Search 输出
            FlaxGenerationMixin,                        # 生成混合类
            FlaxGreedySearchOutput,                     # Flax Greedy Search 输出
            FlaxSampleOutput,                           # Flax Sample 输出
        )
else:
    # 导入 sys 模块，用于动态设置当前模块的引用
    import sys
    # 设置当前模块的引用，将其指向 _LazyModule 实例化的对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```