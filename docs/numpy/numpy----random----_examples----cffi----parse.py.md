# `.\numpy\numpy\random\_examples\cffi\parse.py`

```py
# 导入os模块，用于处理文件路径等操作
import os

# 定义函数parse_distributions_h，接受两个参数ffi和inc_dir
def parse_distributions_h(ffi, inc_dir):
    """
    Parse distributions.h located in inc_dir for CFFI, filling in the ffi.cdef

    Read the function declarations without the "#define ..." macros that will
    be filled in when loading the library.
    """

    # 打开random/bitgen.h文件进行读取，并初始化空列表s
    with open(os.path.join(inc_dir, 'random', 'bitgen.h')) as fid:
        s = []
        # 逐行处理文件内容
        for line in fid:
            # 如果行以'#'开头，则跳过该行（处理注释或预处理指令）
            if line.strip().startswith('#'):
                continue
            # 将不以'#'开头的行加入列表s
            s.append(line)
        # 将列表s中的内容作为字符串连接起来，传递给ffi.cdef以定义接口
        ffi.cdef('\n'.join(s))

    # 打开random/distributions.h文件进行读取，并初始化空列表s，以及标记变量和忽略状态
    with open(os.path.join(inc_dir, 'random', 'distributions.h')) as fid:
        s = []
        in_skip = 0
        ignoring = False
        # 逐行处理文件内容
        for line in fid:
            # 如果当前处于忽略状态，遇到'#endif'则结束忽略状态
            if ignoring:
                if line.strip().startswith('#endif'):
                    ignoring = False
                continue
            # 如果遇到'#ifdef __cplusplus'，则进入忽略状态
            if line.strip().startswith('#ifdef __cplusplus'):
                ignoring = True
            
            # 如果行以'#'开头，则跳过该行（处理注释或预处理指令）
            if line.strip().startswith('#'):
                continue
    
            # 如果行以'static inline'开头，表示可能是内联函数定义，需要跳过
            if line.strip().startswith('static inline'):
                in_skip += line.count('{')  # 统计行内'{'的个数，增加忽略深度
                continue
            elif in_skip > 0:
                in_skip += line.count('{')  # 继续统计行内'{'的个数
                in_skip -= line.count('}')  # 减少统计行内'}'的个数，直到忽略深度为0
                continue
    
            # 替换行内的宏定义，如将'DECLDIR'替换为空字符串，将'RAND_INT_TYPE'替换为'int64_t'
            line = line.replace('DECLDIR', '')
            line = line.replace('RAND_INT_TYPE', 'int64_t')
            # 将处理后的行加入列表s
            s.append(line)
        # 将列表s中的内容作为字符串连接起来，传递给ffi.cdef以定义接口
        ffi.cdef('\n'.join(s))
```