# `D:\src\scipysrc\scipy\scipy\datasets\__init__.py`

```
"""
================================
Datasets (:mod:`scipy.datasets`)
================================

.. currentmodule:: scipy.datasets

Dataset Methods
===============

.. autosummary::
   :toctree: generated/

   ascent
   face
   electrocardiogram

Utility Methods
===============

.. autosummary::
   :toctree: generated/

   download_all    -- Download all the dataset files to specified path.
   clear_cache     -- Clear cached dataset directory.


Usage of Datasets
=================

SciPy dataset methods can be simply called as follows: ``'<dataset-name>()'``
This downloads the dataset files over the network once, and saves the cache,
before returning a `numpy.ndarray` object representing the dataset.

Note that the return data structure and data type might be different for
different dataset methods. For a more detailed example on usage, please look
into the particular dataset method documentation above.


How dataset retrieval and storage works
=======================================

SciPy dataset files are stored within individual github repositories under the
SciPy GitHub organization, following a naming convention as
``'dataset-<name>'``, for example `scipy.datasets.face` files live at
https://github.com/scipy/dataset-face.  The `scipy.datasets` submodule utilizes
and depends on `Pooch <https://www.fatiando.org/pooch/latest/>`_, a Python
package built to simplify fetching data files. Pooch uses these repos to
retrieve the respective dataset files when calling the dataset function.

A registry of all the datasets, essentially a mapping of filenames with their
SHA256 hash and repo urls are maintained, which Pooch uses to handle and verify
the downloads on function call. After downloading the dataset once, the files
are saved in the system cache directory under ``'scipy-data'``.

Dataset cache locations may vary on different platforms.

For macOS::

    '~/Library/Caches/scipy-data'

For Linux and other Unix-like platforms::

    '~/.cache/scipy-data'  # or the value of the XDG_CACHE_HOME env var, if defined

For Windows::

    'C:\\Users\\<user>\\AppData\\Local\\<AppAuthor>\\scipy-data\\Cache'


In environments with constrained network connectivity for various security
reasons or on systems without continuous internet connections, one may manually
load the cache of the datasets by placing the contents of the dataset repo in
the above mentioned cache directory to avoid fetching dataset errors without
the internet connectivity.

"""


# 从 _fetchers 模块导入数据集方法 face, ascent, electrocardiogram
from ._fetchers import face, ascent, electrocardiogram
# 导入下载所有数据集文件的函数 download_all
from ._download_all import download_all
# 导入清除缓存的函数 clear_cache
from ._utils import clear_cache

# 模块的公开接口，包括数据集方法和工具方法
__all__ = ['ascent', 'electrocardiogram', 'face',
           'download_all', 'clear_cache']

# 导入用于测试的 PytestTester 类，并创建一个测试对象 test
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
# 删除 PytestTester 类的引用，以避免污染命名空间
del PytestTester
```