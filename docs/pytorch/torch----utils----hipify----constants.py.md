# `.\pytorch\torch\utils\hipify\constants.py`

```py
# 定义转换版本的常量，值为0
CONV_VERSION = 0,
# 定义初始化转换的常量，值为1
CONV_INIT = 1
# 定义设备转换的常量，值为2
CONV_DEVICE = 2
# 定义内存转换的常量，值为3
CONV_MEM = 3
# 定义内核转换的常量，值为4
CONV_KERN = 4
# 定义坐标函数转换的常量，值为5
CONV_COORD_FUNC = 5
# 定义数学函数转换的常量，值为6
CONV_MATH_FUNC = 6
# 定义设备函数转换的常量，值为7
CONV_DEVICE_FUNC = 7
# 定义特殊函数转换的常量，值为8
CONV_SPECIAL_FUNC = 8
# 定义流转换的常量，值为9
CONV_STREAM = 9
# 定义事件转换的常量，值为10
CONV_EVENT = 10
# 定义占用率转换的常量，值为11
CONV_OCCUPANCY = 11
# 定义上下文转换的常量，值为12
CONV_CONTEXT = 12
# 定义对等转换的常量，值为13
CONV_PEER = 13
# 定义模块转换的常量，值为14
CONV_MODULE = 14
# 定义缓存转换的常量，值为15
CONV_CACHE = 15
# 定义执行转换的常量，值为16
CONV_EXEC = 16
# 定义错误转换的常量，值为17
CONV_ERROR = 17
# 定义默认转换的常量，值为18
CONV_DEF = 18
# 定义纹理转换的常量，值为19
CONV_TEX = 19
# 定义OpenGL转换的常量，值为20
CONV_GL = 20
# 定义图形转换的常量，值为21
CONV_GRAPHICS = 21
# 定义表面转换的常量，值为22
CONV_SURFACE = 22
# 定义即时编译转换的常量，值为23
CONV_JIT = 23
# 定义Direct3D 9转换的常量，值为24
CONV_D3D9 = 24
# 定义Direct3D 10转换的常量，值为25
CONV_D3D10 = 25
# 定义Direct3D 11转换的常量，值为26
CONV_D3D11 = 26
# 定义VDPAU转换的常量，值为27
CONV_VDPAU = 27
# 定义EGL转换的常量，值为28
CONV_EGL = 28
# 定义线程转换的常量，值为29
CONV_THREAD = 29
# 定义其他转换的常量，值为30
CONV_OTHER = 30
# 定义包含转换的常量，值为31
CONV_INCLUDE = 31
# 定义包含CUDA主头文件转换的常量，值为32
CONV_INCLUDE_CUDA_MAIN_H = 32
# 定义类型转换的常量，值为33
CONV_TYPE = 33
# 定义字面量转换的常量，值为34
CONV_LITERAL = 34
# 定义数值字面量转换的常量，值为35
CONV_NUMERIC_LITERAL = 35
# 定义最后一个转换的常量，值为36
CONV_LAST = 36

# 定义驱动API转换的常量，值为37
API_DRIVER = 37
# 定义运行时API转换的常量，值为38
API_RUNTIME = 38
# 定义BLAS API转换的常量，值为39
API_BLAS = 39
# 定义特殊API转换的常量，值为40
API_SPECIAL = 40
# 定义随机数API转换的常量，值为41
API_RAND = 41
# 定义最后一个API转换的常量，值为42
API_LAST = 42
# 定义FFT API转换的常量，值为43
API_FFT = 43
# 定义RTC API转换的常量，值为44
API_RTC = 44
# 定义ROCTX API转换的常量，值为45
API_ROCTX = 45

# 定义不支持的HIP转换的常量，值为46
HIP_UNSUPPORTED = 46
# 定义PyTorch API转换的常量，值为1337
API_PYTORCH = 1337
# 定义Caffe2 API转换的常量，值为1338
API_CAFFE2 = 1338
# 定义C10 API转换的常量，值为1339
API_C10 = 1339
# 定义ROCm SMI API转换的常量，值为1340
API_ROCMSMI = 1340
```