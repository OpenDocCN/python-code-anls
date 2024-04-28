# `.\transformers\models\mgp_str\__init__.py`

```
# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import necessary modules and functions
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# specify the import structure
_import_structure = {
    "configuration_mgp_str": ["MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP", "MgpstrConfig"],
    "processing_mgp_str": ["MgpstrProcessor"],
    "tokenization_mgp_str": ["MgpstrTokenizer"],
}

# check if torch is available, if not raise OptionalDependencyNotAvailable
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # if torch is available, add additional items to the import structure
    _import_structure["modeling_mgp_str"] = [
        "MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MgpstrModel",
        "MgpstrPreTrainedModel",
        "MgpstrForSceneTextRecognition",
    ]

# if TYPE_CHECKING is True, import necessary classes from corresponding modules
if TYPE_CHECKING:
    from .configuration_mgp_str import MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP, MgpstrConfig
    from .processing_mgp_str import MgpstrProcessor
    from .tokenization_mgp_str import MgpstrTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mgp_str import (
            MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST,
            MgpstrForSceneTextRecognition,
            MgpstrModel,
            MgpstrPreTrainedModel,
        )
else:
    # if TYPE_CHECKING is False, import sys and assign _LazyModule to the current module
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```