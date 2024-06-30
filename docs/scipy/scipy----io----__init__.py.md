# `D:\src\scipysrc\scipy\scipy\io\__init__.py`

```
"""
==================================
Input and output (:mod:`scipy.io`)
==================================

.. currentmodule:: scipy.io

SciPy has many modules, classes, and functions available to read data
from and write data to a variety of file formats.

.. seealso:: `NumPy IO routines <https://www.numpy.org/devdocs/reference/routines.io.html>`__

MATLAB® files
=============

.. autosummary::
   :toctree: generated/

   loadmat - Read a MATLAB style mat file (version 4 through 7.1)
   savemat - Write a MATLAB style mat file (version 4 through 7.1)
   whosmat - List contents of a MATLAB style mat file (version 4 through 7.1)

For low-level MATLAB reading and writing utilities, see `scipy.io.matlab`.

IDL® files
==========

.. autosummary::
   :toctree: generated/

   readsav - Read an IDL 'save' file

Matrix Market files
===================

.. autosummary::
   :toctree: generated/

   mminfo - Query matrix info from Matrix Market formatted file
   mmread - Read matrix from Matrix Market formatted file
   mmwrite - Write matrix to Matrix Market formatted file

Unformatted Fortran files
===============================

.. autosummary::
   :toctree: generated/

   FortranFile - A file object for unformatted sequential Fortran files
   FortranEOFError - Exception indicating the end of a well-formed file
   FortranFormattingError - Exception indicating an inappropriate end

Netcdf
======

.. autosummary::
   :toctree: generated/

   netcdf_file - A file object for NetCDF data
   netcdf_variable - A data object for the netcdf module

Harwell-Boeing files
====================

.. autosummary::
   :toctree: generated/

   hb_read   -- read H-B file
   hb_write  -- write H-B file

Wav sound files (:mod:`scipy.io.wavfile`)
=========================================

.. module:: scipy.io.wavfile

.. autosummary::
   :toctree: generated/

   read
   write
   WavFileWarning

Arff files (:mod:`scipy.io.arff`)
=================================

.. module:: scipy.io.arff

.. autosummary::
   :toctree: generated/

   loadarff
   MetaData
   ArffError
   ParseArffError
"""
# 导入 MATLAB 文件读写模块
from .matlab import loadmat, savemat, whosmat

# 导入 netCDF 文件支持模块
from ._netcdf import netcdf_file, netcdf_variable

# 导入 Fortran 文件支持模块
from ._fortran import FortranFile, FortranEOFError, FortranFormattingError

# 导入 Matrix Market 文件支持模块
from ._fast_matrix_market import mminfo, mmread, mmwrite

# 导入 IDL 文件读取模块
from ._idl import readsav

# 导入 Harwell-Boeing 文件读写模块
from ._harwell_boeing import hb_read, hb_write

# 弃用的命名空间，在 v2.0.0 版本将会移除
from . import arff, harwell_boeing, idl, mmio, netcdf, wavfile

# 导入测试工具
from scipy._lib._testutils import PytestTester

# 创建用于当前模块的测试对象
test = PytestTester(__name__)

# 清理命名空间，删除测试工具类
del PytestTester
```