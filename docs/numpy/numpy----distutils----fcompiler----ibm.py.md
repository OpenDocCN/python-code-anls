# `.\numpy\numpy\distutils\fcompiler\ibm.py`

```py
import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
import sys  # 导入系统模块
import subprocess  # 导入子进程管理模块

from numpy.distutils.fcompiler import FCompiler  # 导入 NumPy 的 Fortran 编译器抽象类
from numpy.distutils.exec_command import find_executable  # 导入查找可执行文件的函数
from numpy.distutils.misc_util import make_temp_file  # 导入创建临时文件的函数
from distutils import log  # 导入 distutils 日志模块

compilers = ['IBMFCompiler']  # 定义一个包含单个元素的列表，元素为字符串 'IBMFCompiler'

class IBMFCompiler(FCompiler):
    compiler_type = 'ibm'  # 编译器类型字符串设为 'ibm'
    description = 'IBM XL Fortran Compiler'  # 编译器描述字符串
    version_pattern =  r'(xlf\(1\)\s*|)IBM XL Fortran ((Advanced Edition |)Version |Enterprise Edition V|for AIX, V)(?P<version>[^\s*]*)'
    # 版本号正则表达式模式字符串，匹配 IBM XL Fortran 编译器的版本信息格式

    executables = {
        'version_cmd'  : ["<F77>", "-qversion"],  # 版本命令行的可执行文件及参数列表
        'compiler_f77' : ["xlf"],  # Fortran 77 编译器的可执行文件及参数列表
        'compiler_fix' : ["xlf90", "-qfixed"],  # 固定格式 Fortran 90 编译器的可执行文件及参数列表
        'compiler_f90' : ["xlf90"],  # 自由格式 Fortran 90 编译器的可执行文件及参数列表
        'linker_so'    : ["xlf95"],  # 共享库链接器的可执行文件及参数列表
        'archiver'     : ["ar", "-cr"],  # 静态库创建工具的可执行文件及参数列表
        'ranlib'       : ["ranlib"]  # ranlib 工具的可执行文件及参数列表
        }

    def get_version(self,*args,**kwds):
        version = FCompiler.get_version(self,*args,**kwds)  # 调用父类方法获取版本信息

        if version is None and sys.platform.startswith('aix'):
            # 如果未能获取版本信息并且运行环境为 AIX
            lslpp = find_executable('lslpp')  # 查找 lslpp 可执行文件路径
            xlf = find_executable('xlf')  # 查找 xlf 可执行文件路径
            if os.path.exists(xlf) and os.path.exists(lslpp):
                try:
                    o = subprocess.check_output([lslpp, '-Lc', 'xlfcmp'])  # 执行 lslpp 命令获取 xlfcmp 组件信息
                except (OSError, subprocess.CalledProcessError):
                    pass
                else:
                    m = re.search(r'xlfcmp:(?P<version>\d+([.]\d+)+)', o)  # 使用正则表达式从输出中提取版本号
                    if m: version = m.group('version')

        xlf_dir = '/etc/opt/ibmcmp/xlf'
        if version is None and os.path.isdir(xlf_dir):
            # 如果未能获取版本信息并且 xlf 目录存在
            # 对于 Linux 系统，如果 xlf 输出不包含版本信息（例如 xlf 8.1），则尝试另一种方法
            l = sorted(os.listdir(xlf_dir))  # 列出 xlf 目录下所有文件和目录，并按字母顺序排序
            l.reverse()  # 反转列表顺序
            l = [d for d in l if os.path.isfile(os.path.join(xlf_dir, d, 'xlf.cfg'))]  # 过滤出包含 'xlf.cfg' 文件的目录列表
            if l:
                from distutils.version import LooseVersion
                self.version = version = LooseVersion(l[0])  # 尝试使用第一个目录的版本号作为结果

        return version  # 返回获取的版本号

    def get_flags(self):
        return ['-qextname']  # 返回编译器标志列表，包含参数 '-qextname'

    def get_flags_debug(self):
        return ['-g']  # 返回调试标志列表，包含参数 '-g'
    # 获取链接器选项的函数，根据当前操作系统和编译器版本生成适当的选项列表
    def get_flags_linker_so(self):
        # 初始化选项列表
        opt = []
        # 如果当前操作系统是 macOS
        if sys.platform=='darwin':
            # 添加特定于 macOS 的链接器选项
            opt.append('-Wl,-bundle,-flat_namespace,-undefined,suppress')
        else:
            # 对于其他操作系统，添加通用的共享库链接器选项
            opt.append('-bshared')
        # 获取编译器版本信息
        version = self.get_version(ok_status=[0, 40])
        # 如果获取到了版本信息
        if version is not None:
            # 如果当前操作系统是以 'aix' 开头的系统
            if sys.platform.startswith('aix'):
                # 指定 xlf 配置文件的路径
                xlf_cfg = '/etc/xlf.cfg'
            else:
                # 根据获取的版本信息构建 xlf 配置文件的路径
                xlf_cfg = '/etc/opt/ibmcmp/xlf/%s/xlf.cfg' % version
            # 创建临时文件用于写入新的 xlf 配置信息
            fo, new_cfg = make_temp_file(suffix='_xlf.cfg')
            # 输出日志信息，指示正在创建新的配置文件
            log.info('Creating '+new_cfg)
            # 打开 xlf 配置文件进行读取
            with open(xlf_cfg) as fi:
                # 编译正则表达式，用于匹配 crt1.o 文件路径信息
                crt1_match = re.compile(r'\s*crt\s*=\s*(?P<path>.*)/crt1.o').match
                # 逐行读取 xlf 配置文件内容
                for line in fi:
                    # 尝试匹配 crt1.o 的路径信息
                    m = crt1_match(line)
                    # 如果匹配成功
                    if m:
                        # 在临时文件中写入修改后的 crt 路径信息
                        fo.write('crt = %s/bundle1.o\n' % (m.group('path')))
                    else:
                        # 将未匹配到的行直接写入临时文件
                        fo.write(line)
            # 关闭临时文件
            fo.close()
            # 添加新的链接器选项，指定使用新创建的配置文件
            opt.append('-F'+new_cfg)
        # 返回生成的链接器选项列表
        return opt

    # 获取优化选项的函数，简单地返回一个包含 '-O3' 的列表
    def get_flags_opt(self):
        return ['-O3']
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 从 numpy.distutils 中导入 customized_fcompiler 函数
    from numpy.distutils import customized_fcompiler
    # 设置日志的详细程度为 2（通常表示详细信息）
    log.set_verbosity(2)
    # 打印使用定制化编译器 'ibm' 的版本信息
    print(customized_fcompiler(compiler='ibm').get_version())
```