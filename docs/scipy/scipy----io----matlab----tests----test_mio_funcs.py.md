# `D:\src\scipysrc\scipy\scipy\io\matlab\tests\test_mio_funcs.py`

```
# 导入必要的模块和库
import os.path
import io
from scipy.io.matlab._mio5 import MatFile5Reader

# 定义测试数据路径为当前文件夹下的 'data' 子文件夹
test_data_path = os.path.join(os.path.dirname(__file__), 'data')

# 从 MatFile5Reader 中读取 __function_workspace__ 矩阵的变量
def read_minimat_vars(rdr):
    # 初始化读取操作
    rdr.initialize_read()
    # 创建一个字典，初始化包含一个空列表 '__globals__'
    mdict = {'__globals__': []}
    i = 0
    # 循环直到读取到流的末尾
    while not rdr.end_of_stream():
        # 读取变量的头部信息和下一个位置
        hdr, next_position = rdr.read_var_header()
        # 将变量名解码为字符串，如果为 None 则使用 'None'，并解码为 'latin1' 编码
        name = 'None' if hdr.name is None else hdr.name.decode('latin1')
        # 如果变量名为空字符串，则使用默认格式 'var_数字' 来命名
        if name == '':
            name = 'var_%d' % i
            i += 1
        # 读取变量的数据数组，不进行处理
        res = rdr.read_var_array(hdr, process=False)
        # 调整流的位置到下一个位置
        rdr.mat_stream.seek(next_position)
        # 将变量名和对应的数据数组存储到字典中
        mdict[name] = res
        # 如果变量被标记为全局变量，则将其名字添加到 '__globals__' 列表中
        if hdr.is_global:
            mdict['__globals__'].append(name)
    # 返回包含变量名和对应数据的字典
    return mdict

# 从 MATLAB 格式文件中读取工作空间变量
def read_workspace_vars(fname):
    # 打开文件以二进制读取模式
    fp = open(fname, 'rb')
    # 使用 MatFile5Reader 初始化读取器对象
    rdr = MatFile5Reader(fp, struct_as_record=True)
    # 获取文件中的所有变量
    vars = rdr.get_variables()
    # 从变量中获取 '__function_workspace__' 的内容
    fws = vars['__function_workspace__']
    # 创建一个字节流对象，用于包装 '__function_workspace__' 的数据
    ws_bs = io.BytesIO(fws.tobytes())
    # 调整字节流的位置到第二个字节
    ws_bs.seek(2)
    # 将 MatFile5Reader 对象的流设置为包装后的工作空间字节流
    rdr.mat_stream = ws_bs
    # 猜测字节顺序
    mi = rdr.mat_stream.read(2)
    # 如果以 'IM' 开头，则使用小端序 '<'，否则使用大端序 '>'
    rdr.byte_order = mi == b'IM' and '<' or '>'
    # 读取 4 个字节，预计是字节填充
    rdr.mat_stream.read(4)
    # 读取工作空间变量的内容并存储到字典中
    mdict = read_minimat_vars(rdr)
    # 关闭文件
    fp.close()
    # 返回包含所有工作空间变量的字典
    return mdict

# 测试函数，读取示例文件中的工作空间变量
def test_jottings():
    # 示例文件的路径
    fname = os.path.join(test_data_path, 'parabola.mat')
    # 调用读取工作空间变量的函数
    read_workspace_vars(fname)
```