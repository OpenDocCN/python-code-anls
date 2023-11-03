# ArknightMower源码解析 12

# `packaging/network.py`

这段代码是一个用于在 Python 中执行异步 I/O 操作的库 `aiohttp` 的源代码。它定义了一个名为 `asyncio` 的类，该类被用于编写需要使用 `asyncio` 的高并发 Python 代码。

该代码的作用是告诉用户，这个库是经过 `Python` 社区授权的，允许在遵循 `License`（授权协议）的情况下使用。这个 `License` 就是 `http://www.apache.org/licenses/LICENSE-2.0`，它是 `aiohttp` 库使用的开源许可证。

这段代码的主要目的是让用户在遵循 `License` 的前提下自由地使用 `aiohttp` 库，并且允许用户在需要时通过 `aiohttp` 官网获取 `License` 的详细信息。


```py
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
```

这段代码的作用是下载一个预训练的 PPOCH 模型，并使用 TensorFlow One（原 Google Cloud Run）运行时环境来下载它。下载过程包括以下步骤：

1. 下载预训练的 PPOCH 模型：从预训练模型存储服务（如 Google Cloud Storage）下载预训练模型。
2. 将模型保存到本地文件系统：使用 tarfile 库将 PPOCH 模型打包成一个 tar  archive，并将其保存到指定路径。
3. 下载模型：使用 requests 和 tqdm 库下载 tar archive。
4. 如果下载成功，将下载的文件保存到指定路径：使用 requests 和 file.write() 方法将下载的 PPOCH 模型保存到指定路径。

下载过程中，如果遇到问题，将记录错误并返回 0。


```py
import sys
import tarfile
import requests
# from tqdm import tqdm

from ppocr.utils.logging import get_logger


def download_with_progressbar(url, save_path):
    logger = get_logger()
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 1))
        block_size = 1024  # 1 Kibibyte
        # progress_bar = tqdm(
        #     total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                # progress_bar.update(len(data))
                file.write(data)
        # progress_bar.close()
    else:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)


```

这段代码是一个函数 `maybe_download()`，用于下载某个模型存储目录中的压缩包。

函数接受两个参数 `model_storage_directory` 和 `url`。它首先检查 `model_storage_directory` 中是否存在与 `url` 相同的三个文件名 `.pdiparams`, `.pdiparams.info` 和 `.pdmodel`。如果不存在，它将尝试下载这个压缩包，并将其保存到 `model_storage_directory` 中。

具体实现步骤如下：

1. 检查要下载的压缩包是否已经存在。如果是，将它下载到 `model_storage_directory` 中。
2. 如果要下载的压缩包不存在，使用 `下载_with_progressbar()` 函数下载它，并将其保存到 `model_storage_directory` 中。下载过程中使用了 `tarfile` 库。
3. 使用 `os.makedirs()` 函数创建 `model_storage_directory` 目录，如果目录中已经存在该目录，则忽略此步骤。
4. 调用 `download_with_progressbar()` 函数下载压缩包。
5. 使用 `tarfile.open()` 函数打开下载的压缩包，并使用 `getmembers()` 方法获取所有的成员。
6. 对于每个成员，使用 `tarfile.extractfile()` 函数提取文件到临时目录中，并使用 `open()` 函数以写入模式打开文件，写入数据到 `model_storage_directory` 中的指定文件名中。
7. 最后，删除下载的压缩包，以免对系统造成无用文件。


```py
def maybe_download(model_storage_directory, url):
    # using custom model
    tar_file_name_list = ['.pdiparams', '.pdiparams.info', '.pdmodel']
    if not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdiparams')
    ) or not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdmodel')):
        assert url.endswith('.tar'), 'Only supports tar compressed package'
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        print('download {} to {}'.format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, 'r') as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if member.name.endswith(tar_file_name):
                        filename = 'inference' + tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(
                        os.path.join(model_storage_directory, filename),
                        'wb') as f:
                    f.write(file.read())
        os.remove(tmp_path)


```

这两段代码定义了两个函数：`is_link()` 和 `confirm_model_dir_url()`。

`is_link()` 函数用于检查给定的字符串是否以 "http" 开头，如果不是，则返回 `False`，否则返回 `True`。

`confirm_model_dir_url()` 函数接收两个参数：`model_dir` 和 `default_model_dir`，`default_url`。首先，它使用 `is_link()` 函数检查给定的 `model_dir` 是否为 `None` 或以 "http" 开头的字符串。如果是，它将 `default_url` 赋给 `model_dir`，否则，它将 `default_model_dir` 赋给 `model_dir`，并将 `model_dir` 加入 `default_url` 的路径中。最后，它返回 `model_dir` 和 `default_url`。


```py
def is_link(s):
    return s is not None and s.startswith('http')


def confirm_model_dir_url(model_dir, default_model_dir, default_url):
    url = default_url
    if model_dir is None or is_link(model_dir):
        if is_link(model_dir):
            url = model_dir
        file_name = url.split('/')[-1][:-4]
        model_dir = default_model_dir
        model_dir = os.path.join(model_dir, file_name)
    return model_dir, url

```

# `packaging/paddleocr.py`

这段代码是一个Python脚本，它将解释为`numpy-client`库的依赖包。`numpy-client`库是一个用于Numpy编程的Python库，它允许用户通过Numpy接口访问Numpy数组和函数。

具体来说，这段代码包含以下内容：

1. 版权信息：指出该软件的作者和版权持有者，并说明了允许对该软件进行哪些用途。

2. 许可证信息：说明了该软件的许可证，以及可以在哪些条件下使用该软件。这个许可证允许用户在遵循许可证规定的情况下使用该软件，包括但不限于在公开场合演示、修改和重新分发该软件。

3. 导入语句：引入了`os`模块，这是Python标准库中的一个模块，用于操作系统相关操作。

4. 下面的内容：定义了一个名为`import os`的函数，它用于将`os`模块中的函数和变量导入到当前命名空间中。这个函数通常用于在程序中执行操作系统相关操作，例如获取文件和目录路径，或设置环境变量。


```py
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
```

这段代码的作用是：

1. 导入必要的包：`sys`、`importlib`、`paddle`、`cv2`、`logging`、`numpy`、`pathlib`、`base64`、`io`、` BytesIO`。
2. 将当前目录（即`__file__`所在目录）的路径添加到`sys.path`链中，以便在需要时可以动态添加。
3. 加载由`paddle`创建的一个` paddle.dataset.辛普森dataset`对象。
4. 将`cv2`库和`logging`库设置为输出标准。
5. 在代码块中导入`numpy`库，以便在需要时可以创建一个包含`np.inf`值的` numpy.array`对象。
6. 导入`pathlib`库，以便在需要时可以创建一个` Path`对象。
7. 导入`base64`库，以便在需要时可以创建一个` base64.Binary`对象。
8. 导入` BytesIO`库，以便在需要时可以创建一个` io.BytesIO`对象。
9. 在代码块中创建一个` BytesIO`对象，将`paddle.dataset.辛普森dataset`对象中的内容写入其中。
10. 使用`io.BytesIO`对象的方法`write`将数据写入到输出流（这里是一个` BytesIO`对象）中。
11. 在代码块外面，通过`os.path.join`函数将当前目录的路径添加到`sys.path`链中。

总之，这段代码的主要目的是定义了一个函数，该函数可以加载一个名为`辛普森dataset`的` paddle.dataset.dataset`对象，并将`cv2`库和`logging`库设置为输出标准，以便在需要时可以输出图像数据。


```py
import sys
import importlib

__dir__ = os.path.dirname(__file__)

import paddle

sys.path.append(os.path.join(__dir__, ''))

import cv2
import logging
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
```

这段代码的作用是导入 PIL（Python Imaging Library）中的 Image 模块，以及从自己的项目中导入一些工具模块（通过 '.' 模块访问当前目录下的模块）。此外，还从 ppocr 和 ppstructure 模块中导入相应的函数和类。

具体来说，这段代码主要实现了以下功能：

1. 从 tools 模块中导入 predict_system 函数，用于对图像进行预测，并返回预测结果。
2. 从 ppocr 模块中导入日志类，用于记录 PPCO-R 在运行时的情况。
3. 从 ppstructure 模块中导入日志类、图像文件处理类以及 StructureSystem 和 save_structure_res 函数，用于处理 PPCO-R 训练和预测过程中的日志信息、图像文件以及结构系统结果的导出。
4. 从 tools 模块中导入 draw_ocr 和 str2bool 函数，用于绘制图像中的文本框以及将文本内容转换为布尔值。
5. 从 ppocr 模块中导入 get_image_file_list 函数，用于获取训练集中的图像文件列表。
6. 从 ppstructure 模块中导入 init_args 函数，用于初始化 PPCO-R 训练参数。
7. 从 ppstructure 模块中导入 draw_structure_result 函数，用于绘制 PPCO-R 的结构结果。
8. 从 tools 模块中导入 download_with_progressbar 函数，用于下载预训练的 PPCO-R 模型。
9. 从 tools 模块中导入 is_link 函数，用于检查给定的 URL 是否为链接。
10. 从 tools 模块中导入 check_gpu 函数，用于检查当前系统中是否有可用的 GPU 资源。

总之，这段代码定义了一系列函数和类，用于实现 PPCO-R 在训练和预测过程中的相关任务。


```py
from PIL import Image

tools = importlib.import_module('.', 'tools')
ppocr = importlib.import_module('.', 'ppocr')
ppstructure = importlib.import_module('.', 'ppstructure')

from tools.infer import predict_system
from ppocr.utils.logging import get_logger

logger = get_logger()
from ppocr.utils.utility import check_and_read, get_image_file_list
from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from tools.infer.utility import draw_ocr, str2bool, check_gpu
from ppstructure.utility import init_args, draw_structure_result
from ppstructure.predict_system import StructureSystem, save_structure_res, to_excel

```

The given code is a JSON object that defines the structure of the国务院办公厅扩大开放领域 Horizontalion Accessible Cultural Data - tablets version 2.0 data轻盈韩国进行范围评估测试使用声明 (zh_CN).json.

它包含两个部分：

1.  结构JSON：定义了数据结构，包括文本和图片的 URL，以及字典文件和文本文件的路径。
2.  布局JSON：定义了文本和图片的布局，包括文本和图片的位置，宽度和高度，以及背景颜色。

文本json中的数据来源于韩国进行范围评估测试使用声明，从Horizontalion Accessible Cultural Data的表格数据中获取。该声明的数据被分为两个部分：一部分用于从 Picodet 的 LCNet 模型中获取表格的布局，另一部分用于从 CDLA 模型中获取表格的布局。布局JSON中的所有URL都是来源于Picodet和CDLA的官方文件，这些文件可能需要提前从Horizontalion Accessible Cultural Data的GitHub仓库中获取。


```py
__all__ = [
    'PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result',
    'save_structure_res', 'download_with_progressbar', 'to_excel'
]

SUPPORT_DET_MODEL = ['DB']
VERSION = '2.6.1.3'
SUPPORT_REC_MODEL = ['CRNN', 'SVTR_LCNet']
BASE_DIR = os.path.expanduser("~/.paddleocr/")

DEFAULT_OCR_MODEL_VERSION = 'PP-OCRv3'
SUPPORT_OCR_MODEL_VERSION = ['PP-OCR', 'PP-OCRv2', 'PP-OCRv3']
DEFAULT_STRUCTURE_MODEL_VERSION = 'PP-StructureV2'
SUPPORT_STRUCTURE_MODEL_VERSION = ['PP-Structure', 'PP-StructureV2']
MODEL_URLS = {
    'OCR': {
        'PP-OCRv3': {
            'det': {
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar',
                },
                'en': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
                },
                'ml': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar'
                }
            },
            'rec': {
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                },
                'en': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
                'korean': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/korean_dict.txt'
                },
                'japan': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/japan_dict.txt'
                },
                'chinese_cht': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
                },
                'ta': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ta_dict.txt'
                },
                'te': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/te_dict.txt'
                },
                'ka': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ka_dict.txt'
                },
                'latin': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/latin_dict.txt'
                },
                'arabic': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/arabic_dict.txt'
                },
                'cyrillic': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
                },
                'devanagari': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
                },
            },
            'cls': {
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        },
        'PP-OCRv2': {
            'det': {
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar',
                },
            },
            'rec': {
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                }
            },
            'cls': {
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        },
        'PP-OCR': {
            'det': {
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar',
                },
                'en': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar',
                },
                'structure': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar'
                }
            },
            'rec': {
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                },
                'en': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
                'french': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/french_dict.txt'
                },
                'german': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/german_dict.txt'
                },
                'korean': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/korean_dict.txt'
                },
                'japan': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/japan_dict.txt'
                },
                'chinese_cht': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
                },
                'ta': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ta_dict.txt'
                },
                'te': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/te_dict.txt'
                },
                'ka': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ka_dict.txt'
                },
                'latin': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/latin_dict.txt'
                },
                'arabic': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/arabic_dict.txt'
                },
                'cyrillic': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
                },
                'devanagari': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
                },
                'structure': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_dict.txt'
                }
            },
            'cls': {
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        }
    },
    'STRUCTURE': {
        'PP-Structure': {
            'table': {
                'en': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                }
            }
        },
        'PP-StructureV2': {
            'table': {
                'en': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                },
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict_ch.txt'
                }
            },
            'layout': {
                'en': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar',
                    'dict_path':
                    'ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt'
                },
                'ch': {
                    'url':
                    'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar',
                    'dict_path':
                    'ppocr/utils/dict/layout_dict/layout_cdla_dict.txt'
                }
            }
        }
    }
}


```

这段代码是一个 Python 函数，名为 `parse_args`，它用于解析命令行参数。该函数使用 `argparse` 模块来解析命令行参数。函数的作用是接受一个参数 `mMain`，它是一个布尔值，表示是否输出所有的参数列表。

函数首先创建一个名为 `parser` 的 `argparse.ArgumentParser` 对象，然后使用 `add_argument` 方法来添加参数。这些参数包括：

- `--lang`：语言，可以是 "zh" 或 "en"。
- `--det`：检测，如果设置为真，则检测延迟开始。
- `--rec`：记录，如果设置为真，则记录预览图像。
- `--type`：识别类型，可以是 "ocr" 或 "structure"。
- `--ocr_version`：OCR 版本，可以是 "PP-OCRv3" 或 "PP-OCRv2"。
- `--structure_version`：结构模型版本，可以是 "PP-StructureV2" 或 "PP-Structure"。

函数还定义了一个 `for` 循环，用于遍历所有可执行的动作，如果这些动作的参数域是 `["rec_char_dict_path","table_char_dict_path","layout_dict_path"]`，则设置这些参数的值为 `None`。

最后，函数返回一个名为 `args` 的命名对象，该对象包含 `inference_args_dict`，它是通过 `parser.parse_args` 方法解析出来的参数的命名对象。如果 `mMain` 为真，则函数返回 `args`，否则返回 `argparse.Namespace` 对象。


```py
def parse_args(mMain=True):
    import argparse
    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default='ch')
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')
    parser.add_argument(
        "--ocr_version",
        type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default='PP-OCRv3',
        help='OCR Model version, the current model support list is as follows: '
        '1. PP-OCRv3 Support Chinese and English detection and recognition model, and direction classifier model'
        '2. PP-OCRv2 Support Chinese detection and recognition model. '
        '3. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.'
    )
    parser.add_argument(
        "--structure_version",
        type=str,
        choices=SUPPORT_STRUCTURE_MODEL_VERSION,
        default='PP-StructureV2',
        help='Model version, the current model support list is as follows:'
        ' 1. PP-Structure Support en table structure model.'
        ' 2. PP-StructureV2 Support ch and en table structure model.')

    for action in parser._actions:
        if action.dest in [
                'rec_char_dict_path', 'table_char_dict_path', 'layout_dict_path'
        ]:
            action.default = None
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


```

The given code appears to be a programming language implementation, particularly for an automatic text recognition (OLAP) platform. It appears to use a combination of character sets, such as ASCII, to cover a wide range of characters from different languages.

The code defines several helper functions and objects, including the `charset_map` dictionary, which maps characters to their respective ASCII codes. The `银河语` object appears to be a constant defined in a separate file, `fr_银河语.py`, which is then imported and used in the `main.py` file.

The `lang` variable is initialized to an empty string. The value of the `lang` variable is determined based on the value of the `model_urls` dictionary, which maps the name of the OCR model to its respective language code. If the `lang` value is not found in the `model_urls` dictionary, the default language model is used. The `det_lang` variable is initialized to the same value as the `lang` variable if the `lang` value is "en" or "latin". Otherwise, the default detection language is used.

The `trec_model_path` and `trec_data_path` variables appear to be variables defined in the `trec.py` module. The former variable is used to specify the path to a pre-trained detection model for a given language, while the latter is used to specify the path to a pre-trained data set for that language. These variables could be used to train the OCR model using the pre-trained detection model.


```py
def parse_lang(lang):
    latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
    ]
    arabic_lang = ['ar', 'fa', 'ug', 'ur']
    cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
    ]
    devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    assert lang in MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION][
        'rec'], 'param lang must in {}, but got {}'.format(
            MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'].keys(), lang)
    if lang == "ch":
        det_lang = "ch"
    elif lang == 'structure':
        det_lang = 'structure'
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "ml"
    return lang, det_lang


```

这段代码定义了一个名为 `get_model_config` 的函数，用于根据传入的模型类型、版本和模型类型，返回对应的训练语料库链接。

函数的参数包括四个参数：

- `type`：模型类型，可能的值为 OCR、STRUCTURE 或其他模型类型。
- `version`：模型版本号。
- `model_type`：模型类型，可能的值为 OCR、STRUCTURE 或其他模型类型。
- `lang`：支持的语言，可能的值为 EN、FR、ES 等语言。

函数首先检查传入的参数类型是否正确，然后根据参数类型进行相应的查找和版本控制。如果传入参数不正确或者模型版本不存在，函数将抛出 `NotImplementedError` 并返回。

如果参数类型和版本控制都正确，函数将返回一个模型URL，该URL对应于指定模型类型和版本的训练语料库链接。


```py
def get_model_config(type, version, model_type, lang):
    if type == 'OCR':
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    elif type == 'STRUCTURE':
        DEFAULT_MODEL_VERSION = DEFAULT_STRUCTURE_MODEL_VERSION
    else:
        raise NotImplementedError

    model_urls = MODEL_URLS[type]
    if version not in model_urls:
        version = DEFAULT_MODEL_VERSION
    if model_type not in model_urls[version]:
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error('{} models is not support, we only support {}'.format(
                model_type, model_urls[DEFAULT_MODEL_VERSION].keys()))
            sys.exit(-1)

    if lang not in model_urls[version][model_type]:
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                'lang {} is not support, we only support {} for {} models'.
                format(lang, model_urls[DEFAULT_MODEL_VERSION][model_type].keys(
                ), model_type))
            sys.exit(-1)
    return model_urls[version][model_type][lang]


```

This is a Python function that takes an image file path as input and returns an image object. It performs the following checks:

1. If the image file is a bytes object, it is decoded and then converted to a np.uint8 NumPy array.
2. If the image file is a string, it is downloaded and then passed through the `check_and_read` function.
3. If the downloaded image file is a valid image file (e.g. a JPEG or PNG file), it is passed through the `check_img_function`.

The `check_img_function` takes an image file and returns an image object. It performs the following steps:

1. It is converted to a np.uint8 NumPy array.
2. It is checked if it is a valid image file (JPEG or PNG).
3. If the image file is a valid image file, it is passed through the `download_with_progressbar` function.
4. The downloaded image file is then passed through the `cv2.imdecode` function.
5. The resulting image object is then converted to a cv2.CvTImage object.

Note: This function assumes that the required libraries, such as `cv2` and `PIL`, are already installed in your system.


```py
def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # download net image
        if is_link(img):
            download_with_progressbar(img, 'tmp.jpg')
            img = 'tmp.jpg'
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, 'rb') as f:
                img_str = f.read()
                img = img_decode(img_str)
            if img is None:
                try:
                    buf = BytesIO()
                    image = BytesIO(img_str)
                    im = Image.open(image)
                    rgb = im.convert('RGB')
                    rgb.save(buf, 'jpeg')
                    buf.seek(0)
                    image_bytes = buf.read()
                    data_base64 = str(base64.b64encode(image_bytes),
                                      encoding="utf-8")
                    image_decode = base64.b64decode(data_base64)
                    img_array = np.frombuffer(image_decode, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    logger.error("error in loading image:{}".format(image_file))
                    return None
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


```

This is a Python script that uses multi-scale image detection (MUS) and text detection (MD) to detect text in images. It takes a list of images and a classifier with a pre-trained model for text detection.

The script first checks if the classifier is initialized correctly. If not, it will not use the classifier for


```py
class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, "ocr_version must in {}, but get {}".format(
            SUPPORT_OCR_MODEL_VERSION, params.ocr_version)
        params.use_gpu = check_gpu(params.use_gpu)

        logger.setLevel(logging.ERROR)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, 'whl', 'det', det_lang),
            det_model_config['url'])
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])
        cls_model_config = get_model_config('OCR', params.ocr_version, 'cls',
                                            'ch')
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir,
            os.path.join(BASE_DIR, 'whl', 'cls'), cls_model_config['url'])
        if params.ocr_version == 'PP-OCRv3':
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320"
        # download model if using paddle infer
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])

        logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = params.page_num

    def ocr(self, img, det=True, rec=True, cls=True):
        """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not. If false, only rec will be exec. Default is True
            rec: use text recognition or not. If false, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If true, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
            )

        img = check_img(img)
        # for infer pdf file
        if isinstance(img, list):
            if self.page_num > len(img) or self.page_num == 0:
                self.page_num = len(img)
            imgs = img[:self.page_num]
        else:
            imgs = [img]
        if det and rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                dt_boxes, rec_res, _ = self.__call__(img, cls)
                tmp_res = [[box.tolist(), res]
                           for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                dt_boxes, elapse = self.text_detector(img)
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res


```

如果你有一个 Whl 文件，里面定义了一个表格模型，你可以使用 `TableModel` 类来加载这个模型。`TableModel` 类需要三个参数：模型名称、模型配置和模型文件夹。

首先，你需要在项目中确认你的模型文件夹是否存在。然后，你可以使用 `TableModel` 类来加载模型：

python
from PIL import Image
import os
from typing import Tuple

class TableModel:
   def __init__(self, model_name: str, model_config: Tuple[str, ...], model_file_dir: str):
       self.model_name = model_name
       self.model_config = model_config
       self.model_file_dir = model_file_dir

       # 读取配置文件
       config = self.model_config

       # 初始化字体
       self.font = Image.open(os.path.join(self.model_file_dir, f"{model_name}.ttf")).load()

       # 加载图片
       self.img = Image.open(os.path.join(self.model_file_dir, f"{model_name}.jpg"))

   def __call__(self, img: Image, return_ocr_result_in_table: bool = False, img_idx: int = 0) -> Tuple[str, bool]:
       # 在这里做图像处理
       # ...

       res, _ = super().__call__(img, return_ocr_result_in_table, img_idx)

       return res, _


你需要在 `__init__` 方法中读取模型配置文件，并初始化字体和图片。`__call__` 方法将在做图像处理之后返回处理后的结果。


```py
class PPStructure(StructureSystem):
    def __init__(self, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.structure_version in SUPPORT_STRUCTURE_MODEL_VERSION, "structure_version must in {}, but get {}".format(
            SUPPORT_STRUCTURE_MODEL_VERSION, params.structure_version)
        params.use_gpu = check_gpu(params.use_gpu)
        params.mode = 'structure'

        if not params.show_log:
            logger.setLevel(logging.INFO)
        lang, det_lang = parse_lang(params.lang)
        if lang == 'ch':
            table_lang = 'ch'
        else:
            table_lang = 'en'
        if params.structure_version == 'PP-Structure':
            params.merge_no_span_structure = False

        # init model dir
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, 'whl', 'det', det_lang),
            det_model_config['url'])
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])
        table_model_config = get_model_config(
            'STRUCTURE', params.structure_version, 'table', table_lang)
        params.table_model_dir, table_url = confirm_model_dir_url(
            params.table_model_dir,
            os.path.join(BASE_DIR, 'whl', 'table'), table_model_config['url'])
        layout_model_config = get_model_config(
            'STRUCTURE', params.structure_version, 'layout', lang)
        params.layout_model_dir, layout_url = confirm_model_dir_url(
            params.layout_model_dir,
            os.path.join(BASE_DIR, 'whl', 'layout'), layout_model_config['url'])
        # download model
        maybe_download(params.det_model_dir, det_url)
        maybe_download(params.rec_model_dir, rec_url)
        maybe_download(params.table_model_dir, table_url)
        maybe_download(params.layout_model_dir, layout_url)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])
        if params.table_char_dict_path is None:
            params.table_char_dict_path = str(
                Path(__file__).parent / table_model_config['dict_path'])
        if params.layout_dict_path is None:
            params.layout_dict_path = str(
                Path(__file__).parent / layout_model_config['dict_path'])
        logger.debug(params)
        super().__init__(params)

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        img = check_img(img)
        res, _ = super().__call__(
            img, return_ocr_result_in_table, img_idx=img_idx)
        return res


```

It seems like this is a Python script that performs some sort of image processing and layout recovery.

It has a few functions: `convert_poisson_map_to_pdf`, `create_pdf_image_path`, `save_structures_to_pdf`, `convert_poisson_map_to_doc`, and `create_res_structures`. These functions take in arguments such as `output_directory`, `img_name`, `index`, and `recovery_level`, and return either a file path or a dictionary of image paths and their corresponding processing results.

It also imports several modules: `ppstructure`, `cv2`, `os`, and `numpy`.


```py
def main():
    # for cmd
    args = parse_args(mMain=True)
    image_dir = args.image_dir
    if is_link(image_dir):
        download_with_progressbar(image_dir, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_dir))
        return
    if args.type == 'ocr':
        engine = PaddleOCR(**(args.__dict__))
    elif args.type == 'structure':
        engine = PPStructure(**(args.__dict__))
    else:
        raise NotImplementedError

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        if args.type == 'ocr':
            result = engine.ocr(img_path,
                                det=args.det,
                                rec=args.rec,
                                cls=args.use_angle_cls)
            if result is not None:
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        logger.info(line)
        elif args.type == 'structure':
            img, flag_gif, flag_pdf = check_and_read(img_path)
            if not flag_gif and not flag_pdf:
                img = cv2.imread(img_path)

            if args.recovery and args.use_pdf2docx_api and flag_pdf:
                from pdf2docx.converter import Converter
                docx_file = os.path.join(args.output,
                                         '{}.docx'.format(img_name))
                cv = Converter(img_path)
                cv.convert(docx_file)
                cv.close()
                logger.info('docx save to {}'.format(docx_file))
                continue

            if not flag_pdf:
                if img is None:
                    logger.error("error in loading image:{}".format(img_path))
                    continue
                img_paths = [[img_path, img]]
            else:
                img_paths = []
                for index, pdf_img in enumerate(img):
                    os.makedirs(
                        os.path.join(args.output, img_name), exist_ok=True)
                    pdf_img_path = os.path.join(
                        args.output, img_name,
                        img_name + '_' + str(index) + '.jpg')
                    cv2.imwrite(pdf_img_path, pdf_img)
                    img_paths.append([pdf_img_path, pdf_img])

            all_res = []
            for index, (new_img_path, img) in enumerate(img_paths):
                logger.info('processing {}/{} page:'.format(index + 1,
                                                            len(img_paths)))
                new_img_name = os.path.basename(new_img_path).split('.')[0]
                result = engine(new_img_path, img_idx=index)
                save_structure_res(result, args.output, img_name, index)

                if args.recovery and result != []:
                    from copy import deepcopy
                    from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
                    h, w, _ = img.shape
                    result_cp = deepcopy(result)
                    result_sorted = sorted_layout_boxes(result_cp, w)
                    all_res += result_sorted

            if args.recovery and all_res != []:
                try:
                    from ppstructure.recovery.recovery_to_doc import convert_info_docx
                    convert_info_docx(img, all_res, args.output, img_name)
                except Exception as ex:
                    logger.error(
                        "error in layout recovery image:{}, err msg: {}".format(
                            img_name, ex))
                    continue

            for item in all_res:
                item.pop('img')
                item.pop('res')
                logger.info(item)
            logger.info('result save to {}'.format(args.output))

```

# `ui/auto-imports.d.ts`

It looks like you're trying to compare the different types of controls in Vue.js.

The types.json file is a file that defines the types of the Vue.js objects and functions. It appears to provide information about the different controls that are available in Vue.js, including:

* reactivity: a type of data that automatically updates in response to changes in the application's state
* reactive: a type of data that automatically updates in response to changes in the application's state, but it does not trigger a re-render of the component
* readonly: a type of data that automatically updates in response to changes in the application's state, but it does not trigger a re-render of the component
* ref: a type of input that allows the user to update the value of an element in the DOM
* resolveComponent: a type of function that is used to resolve a component by its name
* shallowReactive: a type of object that is used to create a reactive object with a shallow value
* shallowReadonly: a type of object that is used to create a readonly object with a shallow value
* shallowRef: a type of input that allows the user to update the value of an element in the DOM
* toRaw: a type of function that converts a virtual DOM node to a raw DOM node
* toRef: a type of function that converts a virtual DOM node to a ref
* toRefs: a type of function that converts a virtual DOM node to a ref
* toValue: a type of function that converts a value from a string to a number
* triggerRef: a type of function that is used to create a ref that tracks the DOM node that triggered a previously defined action
* unref: a type of function that is used to create a ref that tracks the previously defined DOM node
* useAttrs: a type of function that allows the user to use an object's properties directly in the component's definition
* useCssModule: a type of function that is used to import CSS modules into a component
* useCssVars: a type of function that is used to use CSS variables in a component
* useDialog: a type of function that is used to show a dialog box in the component
* useLoadingBar: a type of function that is used to show a loading bar in the component
* useMessage: a type of function that is used to show a message in the component
* useNotification: a type of function that is used to show a notification in the component
* useSlots: a type of function that is used to allow the component to use slots for displaying content
* watch: a type of function that is used to track changes in the component's state and automatically updates the view when those changes occur
* watchEffect: a type of function that is used to perform side effects in the component, such as fetching data or updating the DOM
* watchPostEffect: a type of function that is used to perform side effects in the component, such as fetching data or updating the DOM
* watchSyncEffect: a type of function that is used to perform side effects in the component, such as fetching data or updating the DOM in a synchronous way


```py
/* eslint-disable */
/* prettier-ignore */
// @ts-nocheck
// Generated by unplugin-auto-import
export {}
declare global {
  const EffectScope: typeof import('vue')['EffectScope']
  const computed: typeof import('vue')['computed']
  const createApp: typeof import('vue')['createApp']
  const customRef: typeof import('vue')['customRef']
  const defineAsyncComponent: typeof import('vue')['defineAsyncComponent']
  const defineComponent: typeof import('vue')['defineComponent']
  const effectScope: typeof import('vue')['effectScope']
  const getCurrentInstance: typeof import('vue')['getCurrentInstance']
  const getCurrentScope: typeof import('vue')['getCurrentScope']
  const h: typeof import('vue')['h']
  const inject: typeof import('vue')['inject']
  const isProxy: typeof import('vue')['isProxy']
  const isReactive: typeof import('vue')['isReactive']
  const isReadonly: typeof import('vue')['isReadonly']
  const isRef: typeof import('vue')['isRef']
  const markRaw: typeof import('vue')['markRaw']
  const nextTick: typeof import('vue')['nextTick']
  const onActivated: typeof import('vue')['onActivated']
  const onBeforeMount: typeof import('vue')['onBeforeMount']
  const onBeforeUnmount: typeof import('vue')['onBeforeUnmount']
  const onBeforeUpdate: typeof import('vue')['onBeforeUpdate']
  const onDeactivated: typeof import('vue')['onDeactivated']
  const onErrorCaptured: typeof import('vue')['onErrorCaptured']
  const onMounted: typeof import('vue')['onMounted']
  const onRenderTracked: typeof import('vue')['onRenderTracked']
  const onRenderTriggered: typeof import('vue')['onRenderTriggered']
  const onScopeDispose: typeof import('vue')['onScopeDispose']
  const onServerPrefetch: typeof import('vue')['onServerPrefetch']
  const onUnmounted: typeof import('vue')['onUnmounted']
  const onUpdated: typeof import('vue')['onUpdated']
  const provide: typeof import('vue')['provide']
  const reactive: typeof import('vue')['reactive']
  const readonly: typeof import('vue')['readonly']
  const ref: typeof import('vue')['ref']
  const resolveComponent: typeof import('vue')['resolveComponent']
  const shallowReactive: typeof import('vue')['shallowReactive']
  const shallowReadonly: typeof import('vue')['shallowReadonly']
  const shallowRef: typeof import('vue')['shallowRef']
  const toRaw: typeof import('vue')['toRaw']
  const toRef: typeof import('vue')['toRef']
  const toRefs: typeof import('vue')['toRefs']
  const toValue: typeof import('vue')['toValue']
  const triggerRef: typeof import('vue')['triggerRef']
  const unref: typeof import('vue')['unref']
  const useAttrs: typeof import('vue')['useAttrs']
  const useCssModule: typeof import('vue')['useCssModule']
  const useCssVars: typeof import('vue')['useCssVars']
  const useDialog: typeof import('naive-ui')['useDialog']
  const useLoadingBar: typeof import('naive-ui')['useLoadingBar']
  const useMessage: typeof import('naive-ui')['useMessage']
  const useNotification: typeof import('naive-ui')['useNotification']
  const useSlots: typeof import('vue')['useSlots']
  const watch: typeof import('vue')['watch']
  const watchEffect: typeof import('vue')['watchEffect']
  const watchPostEffect: typeof import('vue')['watchPostEffect']
  const watchSyncEffect: typeof import('vue')['watchSyncEffect']
}
```

这段代码是一个 TypeScript 的全局声明，用于声明 Vue 中定义的几种数据结构和类型，包括 Component、ComponentPublicInstance、ComputedRef、InjectionKey、PropType 和 Ref，以及它们的属性和方法。

在这些声明中，使用了 @ts-ignore 注解来告诉 TypeScript 不要将这些声明识别为普通变量，而是直接将其作为类型定义。这样可以避免在后续代码中意外地定义了这些变量。

这个声明的主要作用是告诉 TypeScript 编译器，Vue 中定义了哪些数据结构和类型，以便 TypeScript 能够正确地理解和处理这些类型的变量。


```py
// for type re-export
declare global {
  // @ts-ignore
  export type { Component, ComponentPublicInstance, ComputedRef, InjectionKey, PropType, Ref, VNode } from 'vue'
}

```

# `ui/components.d.ts`

This is a TypeScript interface that defines the default export of the `MaaWeekly` component. The `MaaWeekly` component appears to be a data visualization component that displays a table of data with a specified columns and rows.

The default export of the `MaaWeekly` component includes several types such as `NRecord` and `P` which are defined in the `MaaWeeklyTypeScript.vue` component. It also includes a property `tableData` of type `MaaWeeklyTableData` which is a required property for the component. The `tableData` property is expected to be an array of objects that contain the data for each row in the table.

The `MaaWeekly` component is also expected to have a method `refreshData` which is expected to be a function that updates the `tableData` property with the latest data.


```py
/* eslint-disable */
/* prettier-ignore */
// @ts-nocheck
// Generated by unplugin-vue-components
// Read more: https://github.com/vuejs/core/pull/3399
import '@vue/runtime-core'

export {}

declare module '@vue/runtime-core' {
  export interface GlobalComponents {
    Clue: typeof import('./src/components/Clue.vue')['default']
    Email: typeof import('./src/components/Email.vue')['default']
    HelpText: typeof import('./src/components/HelpText.vue')['default']
    MaaBasic: typeof import('./src/components/MaaBasic.vue')['default']
    MaaRecruit: typeof import('./src/components/MaaRecruit.vue')['default']
    MaaRogue: typeof import('./src/components/MaaRogue.vue')['default']
    MaaWeekly: typeof import('./src/components/MaaWeekly.vue')['default']
    NButton: typeof import('naive-ui')['NButton']
    NCard: typeof import('naive-ui')['NCard']
    NCheckbox: typeof import('naive-ui')['NCheckbox']
    NConfigProvider: typeof import('naive-ui')['NConfigProvider']
    NDialogProvider: typeof import('naive-ui')['NDialogProvider']
    NDivider: typeof import('naive-ui')['NDivider']
    NGlobalStyle: typeof import('naive-ui')['NGlobalStyle']
    NH4: typeof import('naive-ui')['NH4']
    NIcon: typeof import('naive-ui')['NIcon']
    NInput: typeof import('naive-ui')['NInput']
    NInputNumber: typeof import('naive-ui')['NInputNumber']
    NLog: typeof import('naive-ui')['NLog']
    NRadio: typeof import('naive-ui')['NRadio']
    NRadioButton: typeof import('naive-ui')['NRadioButton']
    NRadioGroup: typeof import('naive-ui')['NRadioGroup']
    NSelect: typeof import('naive-ui')['NSelect']
    NSlider: typeof import('naive-ui')['NSlider']
    NSpace: typeof import('naive-ui')['NSpace']
    NTable: typeof import('naive-ui')['NTable']
    NTabPane: typeof import('naive-ui')['NTabPane']
    NTabs: typeof import('naive-ui')['NTabs']
    NTimePicker: typeof import('naive-ui')['NTimePicker']
    NTooltip: typeof import('naive-ui')['NTooltip']
    PlanEditor: typeof import('./src/components/PlanEditor.vue')['default']
  }
}

```

# Mower Web UI

Mower 的新界面。短期目标是 Mower 继续保持桌面应用的形态，界面运行在 WebView 中，取代原来的界面。未来考虑支持在浏览器中运行界面。代码也为在其它网站上展示、编辑 Mower 的排班表提供了可能。

本仓库仅包含前端代码，运行需要后端代码支持。

## 开发

开发时需要分别运行后端和前端。

### 后端

需要 Python 3.8 或 3.9。

后端代码在 [ArkMowers/arknights-mower](https://github.com/ArkMowers/arknights-mower) 仓库的 `dev_shawn` 分支中。

安装依赖：

```pybash
pip install -r requirements.txt
pip install Flask flask-cors flask-sock pywebview
```

运行后端：

```pybash
flask --app server run --port=8000 --reload
```

### 前端

需要 Node.js 16。

安装依赖：

```pybash
npm install
```

运行前端的开发服务器：

```pybash
npm run dev
```

根据输出提示，在浏览器中打开窗口即可。

在开发时，前端默认会访问本地 `8000` 端口以连接后端。可以建立 `.env.development.local` 文件，通过 `VITE_HTTP_URL` 指定连接其它地址。例如连接本地的 5000 端口：

```pyplaintext
VITE_HTTP_URL="http://localhost:5000"
```

## 构建与测试

此时无需运行前端的开发服务器，前端构建生产版本的静态文件：

```pybash
npm run build
```

将生成的 `dist` 文件夹复制到 `arknights-mower` 的目录中。此时运行后端：

```py运行
flask --app server run --port=8000
```

直接在浏览器中打开 <http://localhost:8000>，就能看到前端了；运行 `./webview_ui.py`，也能在 WebView 窗口中看到前端。

## 打包

安装依赖：

```pybash
pip install pyinstaller
```

使用 `pyinstaller` 打包：

```pybash
pyinstaller menu.spec
```

生成的 `mower.exe` 在 `dist` 文件夹中。


# `ui/vite.config.js`

该代码使用了 Vite 开发服务器，并且导入了两个插件：vue 和 AutoImport。vue 是一个用于构建 Vue.js 应用程序的插件，而 AutoImport 是一个用于自动导入 Vue.js 组件的插件。

此外，还导入了 Components 和 NaiveUiResolver。Components 是一个用于创建 Vue.js 组件的插件，而 NaiveUiResolver 是一个用于 Naive UI(一种基于 Vue.js 的用户界面库)的 resolver(即用于处理组件中的局部变量、事件等的模块)。

最后，通过 defineConfig() 函数设置了 Vite 开发服务器的一些配置，包括别名(alias)、模块 resolve(使用 webpack 提供的 resolve 函数)以及将当前目录(即项目根目录)映射到 Vue.js 安装目录(使用 fileURLToPath)。


```py
import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { NaiveUiResolver } from 'unplugin-vue-components/resolvers'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    AutoImport({
      imports: [
        'vue',
        {
          'naive-ui': ['useDialog', 'useMessage', 'useNotification', 'useLoadingBar']
        }
      ]
    }),
    Components({
      resolvers: [NaiveUiResolver()]
    })
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})

```

# `ui/src/main.js`

这段代码使用了Vue和Pinia库来实现了一个Web应用程序。它主要的作用是创建一个Vue应用程序实例，并使用该实例来加载和配置Vue组件。

具体来说，它做了以下几件事情：

1. 引入了axios库，以便在应用程序中使用Axios客户端。
2. 引入了VueAxios组件，以便在Vue组件中使用Axios客户端。
3. 创建了一个Vue应用程序实例，并将其命名为“app”。
4. 使用VueAxios组件将Axios配置项添加到Vue应用程序实例中。
5. 使用Pinia库来管理应用程序的状态。
6. 使用Vue的`createApp`函数来配置和实例化Vue应用程序。
7. 使用`app.mount`方法将Vue应用程序实例挂载到一个HTML元素上，使其可以被访问到 JavaScript 代码中。


```py
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import axios from 'axios'
import VueAxios from 'vue-axios'

const app = createApp(App)
app.use(VueAxios, axios)
app.provide('axios', app.config.globalProperties.axios)
app.use(createPinia())

app.mount('#app')

```

# `ui/src/stores/config.js`

The code you provided is a script for the Android game, `Minecraft`. The script uses the Admin Shell (`adb`) and the Minecraft Forge (`maa_mall`) API to interact with the game. The script is deep-level configured and has a number of features, such as the ability to customize the game with themes, sound effects, and screen effects. The script can also be customized by editing the `build_config` file, which is a configuration file provided by the Minecraft Forge.


```py
import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import axios from 'axios'

export const useConfigStore = defineStore('config', () => {
  const adb = ref('')
  const drone_count_limit = ref(0)
  const drone_room = ref('')
  const enable_party = ref(true)
  const free_blacklist = ref([])
  const maa_adb_path = ref('')
  const maa_enable = ref(false)
  const maa_path = ref('')
  const maa_weekly_plan = ref([])
  const maa_rg_enable = ref(0)
  const mail_enable = ref(false)
  const account = ref('')
  const pass_code = ref('')
  const package_type = ref('official')
  const plan_file = ref('')
  const reload_room = ref('')
  const run_mode = ref(1)
  const run_order_delay = ref(10)
  const start_automatically = ref(false)
  const maa_mall_buy = ref('')
  const maa_mall_blacklist = ref('')
  const shop_list = ref([])
  const maa_gap = ref(false)
  const maa_recruitment_time = ref(false)
  const maa_recruit_only_4 = ref(false)
  const simulator = ref({ name: '', index: -1 })
  const resting_threshold = ref(0.5)
  const theme = ref('light')
  const tap_to_launch_game = ref(false)
  const exit_game_when_idle = ref(true)
  const maa_conn_preset = ref('General')
  const maa_touch_option = ref('maatouch')
  const maa_mall_ignore_blacklist_when_full = ref(false)
  const maa_rg_sleep_min = ref('00:00')
  const maa_rg_sleep_max = ref('00:00')
  const maa_credit_fight = ref(true)
  const maa_rg_theme = ref('Mizuki')
  const rogue = ref({})
  const screenshot = ref(0)
  const mail_subject = ref('')

  async function load_shop() {
    const response = await axios.get(`${import.meta.env.VITE_HTTP_URL}/shop`)
    const mall_list = []
    for (const i of response.data) {
      mall_list.push({
        value: i,
        label: i
      })
    }
    shop_list.value = mall_list
  }

  async function load_config() {
    const response = await axios.get(`${import.meta.env.VITE_HTTP_URL}/conf`)
    adb.value = response.data.adb
    drone_count_limit.value = response.data.drone_count_limit
    drone_room.value = response.data.drone_room
    enable_party.value = response.data.enable_party != 0
    free_blacklist.value =
      response.data.free_blacklist == '' ? [] : response.data.free_blacklist.split(',')
    maa_adb_path.value = response.data.maa_adb_path
    maa_enable.value = response.data.maa_enable != 0
    maa_path.value = response.data.maa_path
    maa_rg_enable.value = response.data.maa_rg_enable == 1
    maa_weekly_plan.value = response.data.maa_weekly_plan
    mail_enable.value = response.data.mail_enable != 0
    account.value = response.data.account
    pass_code.value = response.data.pass_code
    package_type.value = response.data.package_type == 1 ? 'official' : 'bilibili'
    plan_file.value = response.data.planFile
    reload_room.value = response.data.reload_room == '' ? [] : response.data.reload_room.split(',')
    run_mode.value = response.data.run_mode == 2 ? 'orders_only' : 'full'
    run_order_delay.value = response.data.run_order_delay
    start_automatically.value = response.data.start_automatically
    maa_mall_buy.value =
      response.data.maa_mall_buy == '' ? [] : response.data.maa_mall_buy.split(',')
    maa_mall_blacklist.value =
      response.data.maa_mall_blacklist == '' ? [] : response.data.maa_mall_blacklist.split(',')
    maa_gap.value = response.data.maa_gap
    maa_recruitment_time.value = response.data.maa_recruitment_time
    maa_recruit_only_4.value = response.data.maa_recruit_only_4
    simulator.value = response.data.simulator
    resting_threshold.value = response.data.resting_threshold
    theme.value = response.data.theme
    tap_to_launch_game.value = response.data.tap_to_launch_game
    tap_to_launch_game.value.enable = tap_to_launch_game.value.enable ? 'tap' : 'adb'
    exit_game_when_idle.value = response.data.exit_game_when_idle
    maa_conn_preset.value = response.data.maa_conn_preset
    maa_touch_option.value = response.data.maa_touch_option
    maa_mall_ignore_blacklist_when_full.value = response.data.maa_mall_ignore_blacklist_when_full
    maa_rg_sleep_max.value = response.data.maa_rg_sleep_max
    maa_rg_sleep_min.value = response.data.maa_rg_sleep_min
    maa_credit_fight.value = response.data.maa_credit_fight
    maa_rg_theme.value = response.data.maa_rg_theme
    rogue.value = response.data.rogue
    screenshot.value = response.data.screenshot
    mail_subject.value = response.data.mail_subject
  }

  function build_config() {
    return {
      account: account.value,
      adb: adb.value,
      drone_count_limit: drone_count_limit.value,
      drone_room: drone_room.value,
      enable_party: enable_party.value ? 1 : 0,
      free_blacklist: free_blacklist.value.join(','),
      maa_adb_path: maa_adb_path.value,
      maa_enable: maa_enable.value ? 1 : 0,
      maa_path: maa_path.value,
      maa_rg_enable: maa_rg_enable.value ? 1 : 0,
      maa_weekly_plan: maa_weekly_plan.value,
      mail_enable: mail_enable.value ? 1 : 0,
      package_type: package_type.value == 'official' ? 1 : 0,
      pass_code: pass_code.value,
      planFile: plan_file.value,
      reload_room: reload_room.value.join(','),
      run_mode: run_mode.value == 'orders_only' ? 2 : 1,
      run_order_delay: run_order_delay.value,
      start_automatically: start_automatically.value,
      maa_mall_buy: maa_mall_buy.value.join(','),
      maa_mall_blacklist: maa_mall_blacklist.value.join(','),
      maa_gap: maa_gap.value,
      maa_recruitment_time: maa_recruitment_time.value,
      maa_recruit_only_4: maa_recruit_only_4.value,
      simulator: simulator.value,
      theme: theme.value,
      resting_threshold: resting_threshold.value,
      tap_to_launch_game: {
        enable: tap_to_launch_game.value.enable == 'tap',
        x: tap_to_launch_game.value.x,
        y: tap_to_launch_game.value.y
      },
      exit_game_when_idle: exit_game_when_idle.value,
      maa_conn_preset: maa_conn_preset.value,
      maa_touch_option: maa_touch_option.value,
      maa_mall_ignore_blacklist_when_full: maa_mall_ignore_blacklist_when_full.value,
      maa_rg_sleep_max: maa_rg_sleep_max.value,
      maa_rg_sleep_min: maa_rg_sleep_min.value,
      maa_credit_fight: maa_credit_fight.value,
      maa_rg_theme: maa_rg_theme.value,
      rogue: rogue.value,
      screenshot: screenshot.value,
      mail_subject: mail_subject.value
    }
  }

  watch(
    [
      adb,
      drone_count_limit,
      drone_room,
      enable_party,
      free_blacklist,
      maa_adb_path,
      maa_enable,
      maa_path,
      maa_weekly_plan,
      maa_rg_enable,
      mail_enable,
      account,
      pass_code,
      package_type,
      reload_room,
      run_mode,
      run_order_delay,
      start_automatically,
      maa_mall_buy,
      maa_mall_blacklist,
      maa_gap,
      maa_recruitment_time,
      maa_recruit_only_4,
      simulator,
      resting_threshold,
      theme,
      tap_to_launch_game,
      exit_game_when_idle,
      maa_conn_preset,
      maa_touch_option,
      maa_mall_ignore_blacklist_when_full,
      maa_rg_sleep_min,
      maa_rg_sleep_max,
      maa_credit_fight,
      maa_rg_theme,
      rogue,
      screenshot,
      mail_subject
    ],
    () => {
      axios.post(`${import.meta.env.VITE_HTTP_URL}/conf`, build_config())
    },
    { deep: true }
  )

  return {
    adb,
    load_config,
    drone_count_limit,
    drone_room,
    enable_party,
    free_blacklist,
    maa_adb_path,
    maa_enable,
    maa_path,
    maa_rg_enable,
    maa_weekly_plan,
    mail_enable,
    account,
    pass_code,
    package_type,
    plan_file,
    reload_room,
    run_mode,
    run_order_delay,
    start_automatically,
    maa_mall_buy,
    maa_mall_blacklist,
    load_shop,
    shop_list,
    maa_gap,
    maa_recruitment_time,
    maa_recruit_only_4,
    build_config,
    simulator,
    resting_threshold,
    theme,
    tap_to_launch_game,
    exit_game_when_idle,
    maa_conn_preset,
    maa_touch_option,
    maa_mall_ignore_blacklist_when_full,
    maa_rg_sleep_min,
    maa_rg_sleep_max,
    maa_credit_fight,
    maa_rg_theme,
    rogue,
    screenshot,
    mail_subject
  }
})

```