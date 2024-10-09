# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\models\layoutlmv3\__init__.py`

```
# 从 transformers 库导入配置、模型和分词器相关的类
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
# 从 transformers 库导入慢速转快速分词器的转换器
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter

# 从本地配置文件导入 LayoutLMv3Config 配置类
from .configuration_layoutlmv3 import LayoutLMv3Config
# 从本地模型文件导入 LayoutLMv3 的不同模型类
from .modeling_layoutlmv3 import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Model,
)
# 从本地分词器文件导入 LayoutLMv3 的分词器类
from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer
# 从本地快速分词器文件导入 LayoutLMv3 的快速分词器类
from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast


# 注册 LayoutLMv3 的配置和模型类（被注释掉的部分）
#AutoConfig.register("layoutlmv3", LayoutLMv3Config)
#AutoModel.register(LayoutLMv3Config, LayoutLMv3Model)
#AutoModelForTokenClassification.register(LayoutLMv3Config, LayoutLMv3ForTokenClassification)
#AutoModelForQuestionAnswering.register(LayoutLMv3Config, LayoutLMv3ForQuestionAnswering)
#AutoModelForSequenceClassification.register(LayoutLMv3Config, LayoutLMv3ForSequenceClassification)
#AutoTokenizer.register(
#    LayoutLMv3Config, slow_tokenizer_class=LayoutLMv3Tokenizer, fast_tokenizer_class=LayoutLMv3TokenizerFast
#)
# 更新慢速到快速转换器的映射，将 LayoutLMv3Tokenizer 关联到 RobertaConverter
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})
```