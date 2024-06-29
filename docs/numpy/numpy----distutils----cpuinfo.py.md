# `.\numpy\numpy\distutils\cpuinfo.py`

```py
#!/usr/bin/env python3
"""
cpuinfo

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Pearu Peterson

"""

# 导入必要的模块和库
import os                    # 操作系统接口
import platform              # 获取平台信息
import re                    # 正则表达式库
import sys                   # 提供对解释器相关功能的访问
import types                 # Python类型相关的操作
import warnings              # 警告控制

from subprocess import getstatusoutput   # 获取命令行输出状态和内容


def getoutput(cmd, successful_status=(0,), stacklevel=1):
    # 尝试执行命令并获取输出和状态
    try:
        status, output = getstatusoutput(cmd)
    except OSError as e:
        # 捕获操作系统异常并发出警告
        warnings.warn(str(e), UserWarning, stacklevel=stacklevel)
        return False, ""
    # 检查状态是否在成功状态列表中，并返回结果和输出内容
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) in successful_status:
        return True, output
    return False, output

def command_info(successful_status=(0,), stacklevel=1, **kw):
    # 执行一组命令并收集输出信息
    info = {}
    for key in kw:
        ok, output = getoutput(kw[key], successful_status=successful_status,
                               stacklevel=stacklevel+1)
        if ok:
            info[key] = output.strip()
    return info

def command_by_line(cmd, successful_status=(0,), stacklevel=1):
    # 执行命令并逐行获取输出内容
    ok, output = getoutput(cmd, successful_status=successful_status,
                           stacklevel=stacklevel+1)
    if not ok:
        return
    for line in output.splitlines():
        yield line.strip()

def key_value_from_command(cmd, sep, successful_status=(0,),
                           stacklevel=1):
    # 从命令输出中获取键值对信息
    d = {}
    for line in command_by_line(cmd, successful_status=successful_status,
                                stacklevel=stacklevel+1):
        l = [s.strip() for s in line.split(sep, 1)]
        if len(l) == 2:
            d[l[0]] = l[1]
    return d

class CPUInfoBase:
    """Holds CPU information and provides methods for requiring
    the availability of various CPU features.
    """

    def _try_call(self, func):
        # 尝试调用给定函数，捕获任何异常
        try:
            return func()
        except Exception:
            pass

    def __getattr__(self, name):
        # 获取未定义的属性时的默认处理方法
        if not name.startswith('_'):
            if hasattr(self, '_'+name):
                attr = getattr(self, '_'+name)
                if isinstance(attr, types.MethodType):
                    return lambda func=self._try_call,attr=attr : func(attr)
            else:
                return lambda : None
        raise AttributeError(name)

    def _getNCPUs(self):
        # 返回 CPU 数量（这里默认为1）
        return 1

    def __get_nbits(self):
        # 获取系统位数（32位或64位）
        abits = platform.architecture()[0]
        nbits = re.compile(r'(\d+)bit').search(abits).group(1)
        return nbits

    def _is_32bit(self):
        # 检查系统是否为32位
        return self.__get_nbits() == '32'

    def _is_64bit(self):
        # 检查系统是否为64位
        return self.__get_nbits() == '64'

class LinuxCPUInfo(CPUInfoBase):
    # Linux 特定的 CPU 信息类
    info = None
    # 初始化方法，用于初始化实例的信息
    def __init__(self):
        # 如果实例的信息已经存在，则直接返回，不执行后续代码
        if self.info is not None:
            return
        # 初始化一个空的信息列表
        info = [ {} ]
        # 获取系统命令 'uname -m' 的输出
        ok, output = getoutput('uname -m')
        # 如果获取成功
        if ok:
            # 将获取的输出去除首尾空白字符，存入info[0]的键 'uname_m' 中
            info[0]['uname_m'] = output.strip()
        try:
            # 尝试打开 '/proc/cpuinfo' 文件
            fo = open('/proc/cpuinfo')
        except OSError as e:
            # 如果打开文件失败，发出警告并抛出异常信息
            warnings.warn(str(e), UserWarning, stacklevel=2)
        else:
            # 如果打开文件成功，遍历文件的每一行
            for line in fo:
                # 分割每行内容，得到键值对列表，去除首尾空白字符
                name_value = [s.strip() for s in line.split(':', 1)]
                # 如果分割后的列表不是长度为2，跳过当前循环
                if len(name_value) != 2:
                    continue
                # 分别取出键和值
                name, value = name_value
                # 如果info为空或者name已经在info[-1]中存在，说明是下一个处理器信息
                if not info or name in info[-1]: # next processor
                    info.append({})
                # 将当前处理器的键值对存入info[-1]中
                info[-1][name] = value
            # 关闭文件流
            fo.close()
        # 将处理后的信息列表赋值给当前实例的类属性info
        self.__class__.info = info

    # 未实现的方法，占位用
    def _not_impl(self): pass

    # 判断当前处理器是否为AMD系列
    def _is_AMD(self):
        return self.info[0]['vendor_id']=='AuthenticAMD'

    # 判断当前处理器是否为AMD Athlon K6-2
    def _is_AthlonK6_2(self):
        return self._is_AMD() and self.info[0]['model'] == '2'

    # 判断当前处理器是否为AMD Athlon K6-3
    def _is_AthlonK6_3(self):
        return self._is_AMD() and self.info[0]['model'] == '3'

    # 判断当前处理器是否为AMD K6系列
    def _is_AthlonK6(self):
        return re.match(r'.*?AMD-K6', self.info[0]['model name']) is not None

    # 判断当前处理器是否为AMD K7系列
    def _is_AthlonK7(self):
        return re.match(r'.*?AMD-K7', self.info[0]['model name']) is not None

    # 判断当前处理器是否为AMD Athlon MP
    def _is_AthlonMP(self):
        return re.match(r'.*?Athlon\(tm\) MP\b',
                        self.info[0]['model name']) is not None

    # 判断当前处理器是否为AMD 64位处理器
    def _is_AMD64(self):
        return self.is_AMD() and self.info[0]['family'] == '15'

    # 判断当前处理器是否为AMD Athlon 64
    def _is_Athlon64(self):
        return re.match(r'.*?Athlon\(tm\) 64\b',
                        self.info[0]['model name']) is not None

    # 判断当前处理器是否为AMD Athlon HX
    def _is_AthlonHX(self):
        return re.match(r'.*?Athlon HX\b',
                        self.info[0]['model name']) is not None

    # 判断当前处理器是否为AMD Opteron
    def _is_Opteron(self):
        return re.match(r'.*?Opteron\b',
                        self.info[0]['model name']) is not None

    # 判断当前处理器是否为AMD Hammer
    def _is_Hammer(self):
        return re.match(r'.*?Hammer\b',
                        self.info[0]['model name']) is not None

    # 判断当前处理器是否为Alpha系列
    def _is_Alpha(self):
        return self.info[0]['cpu']=='Alpha'

    # 判断当前处理器是否为Alpha EV4
    def _is_EV4(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV4'

    # 判断当前处理器是否为Alpha EV5
    def _is_EV5(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV5'

    # 判断当前处理器是否为Alpha EV56
    def _is_EV56(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV56'

    # 判断当前处理器是否为Alpha PCA56
    def _is_PCA56(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'PCA56'

    # 判断当前处理器是否为Intel系列
    def _is_Intel(self):
        return self.info[0]['vendor_id']=='GenuineIntel'

    # 判断当前处理器是否为Intel i486
    def _is_i486(self):
        return self.info[0]['cpu']=='i486'

    # 判断当前处理器是否为Intel i586
    def _is_i586(self):
        return self.is_Intel() and self.info[0]['cpu family'] == '5'

    # 判断当前处理器是否为Intel i686
    def _is_i686(self):
        return self.is_Intel() and self.info[0]['cpu family'] == '6'
    # 判断CPU型号是否包含'Celeron'
    def _is_Celeron(self):
        return re.match(r'.*?Celeron',
                        self.info[0]['model name']) is not None
    
    # 判断CPU型号是否包含'Pentium'
    def _is_Pentium(self):
        return re.match(r'.*?Pentium',
                        self.info[0]['model name']) is not None
    
    # 判断CPU型号是否包含'Pentium'和'II'
    def _is_PentiumII(self):
        return re.match(r'.*?Pentium.*?II\b',
                        self.info[0]['model name']) is not None
    
    # 判断CPU型号是否为'PentiumPro'
    def _is_PentiumPro(self):
        return re.match(r'.*?PentiumPro\b',
                        self.info[0]['model name']) is not None
    
    # 判断CPU型号是否包含'Pentium'和'MMX'
    def _is_PentiumMMX(self):
        return re.match(r'.*?Pentium.*?MMX\b',
                        self.info[0]['model name']) is not None
    
    # 判断CPU型号是否包含'Pentium'和'III'
    def _is_PentiumIII(self):
        return re.match(r'.*?Pentium.*?III\b',
                        self.info[0]['model name']) is not None
    
    # 判断CPU型号是否包含'Pentium'和'IV'或者'4'
    def _is_PentiumIV(self):
        return re.match(r'.*?Pentium.*?(IV|4)\b',
                        self.info[0]['model name']) is not None
    
    # 判断CPU型号是否包含'Pentium'和'M'
    def _is_PentiumM(self):
        return re.match(r'.*?Pentium.*?M\b',
                        self.info[0]['model name']) is not None
    
    # 判断CPU型号是否为'Prescott'，即为Pentium IV且具有SSE3指令集
    def _is_Prescott(self):
        return (self.is_PentiumIV() and self.has_sse3())
    
    # 判断CPU型号是否为'Nocona'，即为Intel系列、特定CPU family、具有SSE3但不具有SSSE3并包含'lm'标志
    def _is_Nocona(self):
        return (self.is_Intel()
                and (self.info[0]['cpu family'] == '6'
                     or self.info[0]['cpu family'] == '15')
                and (self.has_sse3() and not self.has_ssse3())
                and re.match(r'.*?\blm\b', self.info[0]['flags']) is not None)
    
    # 判断CPU型号是否为'Core2'，即为64位、Intel系列且包含'Core(TM)2'标志
    def _is_Core2(self):
        return (self.is_64bit() and self.is_Intel() and
                re.match(r'.*?Core\(TM\)2\b',
                         self.info[0]['model name']) is not None)
    
    # 判断CPU型号是否包含'Itanium'
    def _is_Itanium(self):
        return re.match(r'.*?Itanium\b',
                        self.info[0]['family']) is not None
    
    # 判断CPU型号是否包含'XEON'（不区分大小写）
    def _is_XEON(self):
        return re.match(r'.*?XEON\b',
                        self.info[0]['model name'], re.IGNORECASE) is not None
    
    # _is_Xeon 是 _is_XEON 的别名
    
    # 判断是否只有一个CPU
    def _is_singleCPU(self):
        return len(self.info) == 1
    
    # 获取CPU数量
    def _getNCPUs(self):
        return len(self.info)
    
    # 判断CPU是否具有fdiv错误
    def _has_fdiv_bug(self):
        return self.info[0]['fdiv_bug']=='yes'
    
    # 判断CPU是否具有f00f错误
    def _has_f00f_bug(self):
        return self.info[0]['f00f_bug']=='yes'
    
    # 判断CPU是否支持MMX指令集
    def _has_mmx(self):
        return re.match(r'.*?\bmmx\b', self.info[0]['flags']) is not None
    
    # 判断CPU是否支持SSE指令集
    def _has_sse(self):
        return re.match(r'.*?\bsse\b', self.info[0]['flags']) is not None
    
    # 判断CPU是否支持SSE2指令集
    def _has_sse2(self):
        return re.match(r'.*?\bsse2\b', self.info[0]['flags']) is not None
    
    # 判断CPU是否支持SSE3指令集
    def _has_sse3(self):
        return re.match(r'.*?\bpni\b', self.info[0]['flags']) is not None
    
    # 判断CPU是否支持SSSE3指令集
    def _has_ssse3(self):
        return re.match(r'.*?\bssse3\b', self.info[0]['flags']) is not None
    
    # 判断CPU是否支持3DNow指令集
    def _has_3dnow(self):
        return re.match(r'.*?\b3dnow\b', self.info[0]['flags']) is not None
    
    # 判断CPU是否支持3DNowExt指令集
    def _has_3dnowext(self):
        return re.match(r'.*?\b3dnowext\b', self.info[0]['flags']) is not None
class IRIXCPUInfo(CPUInfoBase):
    # IRIXCPUInfo 类，继承自 CPUInfoBase

    info = None
    # 类变量 info，用于存储系统信息，初始值为 None

    def __init__(self):
        # 初始化方法
        if self.info is not None:
            return
        # 如果 info 已经被设置过，则直接返回，避免重复初始化

        # 调用 key_value_from_command 函数获取系统信息，命令为 'sysconf'，分隔符为 ' '
        # 只接受状态码为 0 或 1 的执行结果
        info = key_value_from_command('sysconf', sep=' ',
                                      successful_status=(0, 1))
        # 设置类变量 info 为获取到的系统信息
        self.__class__.info = info

    def _not_impl(self): pass
    # 占位方法，未实现的方法，不做任何操作

    def _is_singleCPU(self):
        # 判断是否为单 CPU 系统，根据 info 中的 'NUM_PROCESSORS' 值来判断
        return self.info.get('NUM_PROCESSORS') == '1'

    def _getNCPUs(self):
        # 获取 CPU 数量，如果 'NUM_PROCESSORS' 存在则返回其整数值，否则返回默认值 1
        return int(self.info.get('NUM_PROCESSORS', 1))

    def __cputype(self, n):
        # 辅助方法，用于判断 CPU 类型是否为指定编号 n
        return self.info.get('PROCESSORS').split()[0].lower() == 'r%s' % (n)

    def _is_r2000(self): return self.__cputype(2000)
    def _is_r3000(self): return self.__cputype(3000)
    def _is_r3900(self): return self.__cputype(3900)
    def _is_r4000(self): return self.__cputype(4000)
    def _is_r4100(self): return self.__cputype(4100)
    def _is_r4300(self): return self.__cputype(4300)
    def _is_r4400(self): return self.__cputype(4400)
    def _is_r4600(self): return self.__cputype(4600)
    def _is_r4650(self): return self.__cputype(4650)
    def _is_r5000(self): return self.__cputype(5000)
    def _is_r6000(self): return self.__cputype(6000)
    def _is_r8000(self): return self.__cputype(8000)
    def _is_r10000(self): return self.__cputype(10000)
    def _is_r12000(self): return self.__cputype(12000)
    def _is_rorion(self): return self.__cputype('orion')

    def get_ip(self):
        # 获取主机名（IP），如果出错则忽略异常
        try: return self.info.get('MACHINE')
        except Exception: pass

    def __machine(self, n):
        # 辅助方法，用于判断主机类型是否为指定编号 n
        return self.info.get('MACHINE').lower() == 'ip%s' % (n)

    def _is_IP19(self): return self.__machine(19)
    def _is_IP20(self): return self.__machine(20)
    def _is_IP21(self): return self.__machine(21)
    def _is_IP22(self): return self.__machine(22)
    def _is_IP22_4k(self): return self.__machine(22) and self._is_r4000()
    def _is_IP22_5k(self): return self.__machine(22) and self._is_r5000()
    def _is_IP24(self): return self.__machine(24)
    def _is_IP25(self): return self.__machine(25)
    def _is_IP26(self): return self.__machine(26)
    def _is_IP27(self): return self.__machine(27)
    def _is_IP28(self): return self.__machine(28)
    def _is_IP30(self): return self.__machine(30)
    def _is_IP32(self): return self.__machine(32)
    def _is_IP32_5k(self): return self.__machine(32) and self._is_r5000()
    def _is_IP32_10k(self): return self.__machine(32) and self._is_r10000()


class DarwinCPUInfo(CPUInfoBase):
    # DarwinCPUInfo 类，继承自 CPUInfoBase

    info = None
    # 类变量 info，用于存储系统信息，初始值为 None

    def __init__(self):
        # 初始化方法
        if self.info is not None:
            return
        # 如果 info 已经被设置过，则直接返回，避免重复初始化

        # 调用 command_info 函数获取系统信息，包括 'arch' 和 'machine' 的命令结果
        info = command_info(arch='arch',
                            machine='machine')
        # 使用 sysctl 命令获取更多硬件相关信息，将其存储在 'sysctl_hw' 键下
        info['sysctl_hw'] = key_value_from_command('sysctl hw', sep='=')
        # 设置类变量 info 为获取到的所有系统信息
        self.__class__.info = info

    def _not_impl(self): pass
    # 占位方法，未实现的方法，不做任何操作

    def _getNCPUs(self):
        # 获取 CPU 数量，从 sysctl_hw 字典中获取 'hw.ncpu' 键值，如果不存在则返回默认值 1
        return int(self.info['sysctl_hw'].get('hw.ncpu', 1))

    def _is_Power_Macintosh(self):
        # 判断当前系统是否为 Power Macintosh
        return self.info['sysctl_hw']['hw.machine']=='Power Macintosh'
    # 判断当前系统架构是否为 i386
    def _is_i386(self):
        return self.info['arch']=='i386'
    
    # 判断当前系统架构是否为 ppc
    def _is_ppc(self):
        return self.info['arch']=='ppc'
    
    # 检查当前系统的具体机器型号是否为给定的 PPC 架构类型
    def __machine(self, n):
        return self.info['machine'] == 'ppc%s'%n
    
    # 下面一系列方法用于检查当前系统是否为特定的 PPC 机器型号
    def _is_ppc601(self): return self.__machine(601)
    def _is_ppc602(self): return self.__machine(602)
    def _is_ppc603(self): return self.__machine(603)
    def _is_ppc603e(self): return self.__machine('603e')
    def _is_ppc604(self): return self.__machine(604)
    def _is_ppc604e(self): return self.__machine('604e')
    def _is_ppc620(self): return self.__machine(620)
    def _is_ppc630(self): return self.__machine(630)
    def _is_ppc740(self): return self.__machine(740)
    def _is_ppc7400(self): return self.__machine(7400)
    def _is_ppc7450(self): return self.__machine(7450)
    def _is_ppc750(self): return self.__machine(750)
    def _is_ppc403(self): return self.__machine(403)
    def _is_ppc505(self): return self.__machine(505)
    def _is_ppc801(self): return self.__machine(801)
    def _is_ppc821(self): return self.__machine(821)
    def _is_ppc823(self): return self.__machine(823)
    def _is_ppc860(self): return self.__machine(860)
# 定义一个名为 SunOSCPUInfo 的类，继承自 CPUInfoBase 类
class SunOSCPUInfo(CPUInfoBase):

    # 类变量 info，用于存储系统信息，初始化为 None
    info = None

    # 构造函数，初始化对象实例
    def __init__(self):
        # 如果类变量 info 不为 None，则直接返回，避免重复初始化
        if self.info is not None:
            return
        
        # 调用 command_info 函数获取系统信息，包括架构、机器类型、内核信息等
        info = command_info(arch='arch',
                            mach='mach',
                            uname_i='uname_i',
                            isainfo_b='isainfo -b',
                            isainfo_n='isainfo -n',
                            )
        
        # 调用 key_value_from_command 函数获取并添加额外的 uname -X 命令返回的键值对信息到 info 字典中
        info['uname_X'] = key_value_from_command('uname -X', sep='=')
        
        # 通过迭代 psrinfo -v 0 命令返回的每一行，使用正则表达式匹配处理器信息，并将结果添加到 info 字典中
        for line in command_by_line('psrinfo -v 0'):
            m = re.match(r'\s*The (?P<p>[\w\d]+) processor operates at', line)
            if m:
                info['processor'] = m.group('p')
                break
        
        # 将当前实例的类变量 info 设置为获取到的系统信息
        self.__class__.info = info

    # 以下是一系列用于检测特定硬件或软件特征的方法，每个方法都返回布尔值

    # 检测系统是否为 i386 架构
    def _is_i386(self):
        return self.info['isainfo_n']=='i386'

    # 检测系统是否为 sparc 架构
    def _is_sparc(self):
        return self.info['isainfo_n']=='sparc'

    # 检测系统是否为 sparcv9 架构
    def _is_sparcv9(self):
        return self.info['isainfo_n']=='sparcv9'

    # 获取系统中 CPU 的数量，如果未指定则默认为 1
    def _getNCPUs(self):
        return int(self.info['uname_X'].get('NumCPU', 1))

    # 检测系统是否为 sun4 架构
    def _is_sun4(self):
        return self.info['arch']=='sun4'

    # 检测 uname_i 是否以 SUNW 开头，表示系统是否为 SUNW 类型
    def _is_SUNW(self):
        return re.match(r'SUNW', self.info['uname_i']) is not None

    # 检测 uname_i 是否包含字符串 SPARCstation-5，表示系统是否为 SPARCstation-5 型号
    def _is_sparcstation5(self):
        return re.match(r'.*SPARCstation-5', self.info['uname_i']) is not None

    # 检测 uname_i 是否包含字符串 Ultra-1，表示系统是否为 Ultra-1 型号
    def _is_ultra1(self):
        return re.match(r'.*Ultra-1', self.info['uname_i']) is not None

    # 检测 uname_i 是否包含字符串 Ultra-250，表示系统是否为 Ultra-250 型号
    def _is_ultra250(self):
        return re.match(r'.*Ultra-250', self.info['uname_i']) is not None

    # 以下方法类似，用于检测其他型号
    def _is_ultra2(self):
        return re.match(r'.*Ultra-2', self.info['uname_i']) is not None

    def _is_ultra30(self):
        return re.match(r'.*Ultra-30', self.info['uname_i']) is not None

    def _is_ultra4(self):
        return re.match(r'.*Ultra-4', self.info['uname_i']) is not None

    def _is_ultra5_10(self):
        return re.match(r'.*Ultra-5_10', self.info['uname_i']) is not None

    def _is_ultra5(self):
        return re.match(r'.*Ultra-5', self.info['uname_i']) is not None

    def _is_ultra60(self):
        return re.match(r'.*Ultra-60', self.info['uname_i']) is not None

    def _is_ultra80(self):
        return re.match(r'.*Ultra-80', self.info['uname_i']) is not None

    def _is_ultraenterprice(self):
        return re.match(r'.*Ultra-Enterprise', self.info['uname_i']) is not None

    def _is_ultraenterprice10k(self):
        return re.match(r'.*Ultra-Enterprise-10000', self.info['uname_i']) is not None

    def _is_sunfire(self):
        return re.match(r'.*Sun-Fire', self.info['uname_i']) is not None

    def _is_ultra(self):
        return re.match(r'.*Ultra', self.info['uname_i']) is not None

    # 检测当前处理器是否为 sparcv7 架构
    def _is_cpusparcv7(self):
        return self.info['processor']=='sparcv7'

    # 检测当前处理器是否为 sparcv8 架构
    def _is_cpusparcv8(self):
        return self.info['processor']=='sparcv8'

    # 检测当前处理器是否为 sparcv9 架构
    def _is_cpusparcv9(self):
        return self.info['processor']=='sparcv9'


# 定义一个名为 Win32CPUInfo 的类，继承自 CPUInfoBase 类
class Win32CPUInfo(CPUInfoBase):

    # 类变量 info，用于存储系统信息，初始化为 None
    info = None
    pkey = r"HARDWARE\DESCRIPTION\System\CentralProcessor"
    # 定义注册表路径，指定到处理器信息的位置

    # 构造函数初始化
    def __init__(self):
        if self.info is not None:
            return
        info = []
        try:
            # 尝试导入winreg模块，用于访问Windows注册表
            import winreg

            # 正则表达式匹配处理器标识信息的模式
            prgx = re.compile(r"family\s+(?P<FML>\d+)\s+model\s+(?P<MDL>\d+)"
                              r"\s+stepping\s+(?P<STP>\d+)", re.IGNORECASE)
            # 打开注册表指定路径下的键
            chnd=winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.pkey)
            pnum=0
            # 遍历处理器子键
            while True:
                try:
                    proc=winreg.EnumKey(chnd, pnum)
                except winreg.error:
                    break
                else:
                    pnum+=1
                    # 向信息列表添加处理器名称
                    info.append({"Processor":proc})
                    # 打开处理器子键
                    phnd=winreg.OpenKey(chnd, proc)
                    pidx=0
                    # 遍历处理器子键中的值
                    while True:
                        try:
                            name, value, vtpe=winreg.EnumValue(phnd, pidx)
                        except winreg.error:
                            break
                        else:
                            pidx=pidx+1
                            # 将处理器属性及其值添加到信息列表最后一个字典中
                            info[-1][name]=value
                            # 如果属性名为"Identifier"，则使用正则表达式解析处理器标识信息
                            if name=="Identifier":
                                srch=prgx.search(value)
                                if srch:
                                    info[-1]["Family"]=int(srch.group("FML"))
                                    info[-1]["Model"]=int(srch.group("MDL"))
                                    info[-1]["Stepping"]=int(srch.group("STP"))
        except Exception as e:
            # 捕获异常并打印错误信息，忽略处理器信息的获取错误
            print(e, '(ignoring)')
        # 将获取的处理器信息赋给类属性info
        self.__class__.info = info

    # 未实现的方法，占位符方法
    def _not_impl(self): pass

    # Athlon处理器判断方法
    def _is_AMD(self):
        return self.info[0]['VendorIdentifier']=='AuthenticAMD'

    # Am486处理器判断方法
    def _is_Am486(self):
        return self.is_AMD() and self.info[0]['Family']==4

    # Am5x86处理器判断方法
    def _is_Am5x86(self):
        return self.is_AMD() and self.info[0]['Family']==4

    # AMDK5处理器判断方法
    def _is_AMDK5(self):
        return self.is_AMD() and self.info[0]['Family']==5 \
               and self.info[0]['Model'] in [0, 1, 2, 3]

    # AMDK6处理器判断方法
    def _is_AMDK6(self):
        return self.is_AMD() and self.info[0]['Family']==5 \
               and self.info[0]['Model'] in [6, 7]

    # AMDK6_2处理器判断方法
    def _is_AMDK6_2(self):
        return self.is_AMD() and self.info[0]['Family']==5 \
               and self.info[0]['Model']==8

    # AMDK6_3处理器判断方法
    def _is_AMDK6_3(self):
        return self.is_AMD() and self.info[0]['Family']==5 \
               and self.info[0]['Model']==9

    # AMDK7处理器判断方法
    def _is_AMDK7(self):
        return self.is_AMD() and self.info[0]['Family'] == 6

    # 要可靠地区分不同类型的AMD64芯片（如Athlon64、Operton、Athlon64 X2、Semperon、Turion 64等）
    # 需要查看cpuid的“brand”信息
    # 判断当前 CPU 是否为 AMD64 架构
    def _is_AMD64(self):
        return self.is_AMD() and self.info[0]['Family'] == 15

    # 判断当前 CPU 是否为 Intel 厂商
    def _is_Intel(self):
        return self.info[0]['VendorIdentifier']=='GenuineIntel'

    # 判断当前 CPU 是否为 Intel 386 系列
    def _is_i386(self):
        return self.info[0]['Family']==3

    # 判断当前 CPU 是否为 Intel 486 系列
    def _is_i486(self):
        return self.info[0]['Family']==4

    # 判断当前 CPU 是否为 Intel 586 系列
    def _is_i586(self):
        return self.is_Intel() and self.info[0]['Family']==5

    # 判断当前 CPU 是否为 Intel 686 系列
    def _is_i686(self):
        return self.is_Intel() and self.info[0]['Family']==6

    # 判断当前 CPU 是否为 Pentium 系列
    def _is_Pentium(self):
        return self.is_Intel() and self.info[0]['Family']==5

    # 判断当前 CPU 是否为 Pentium MMX
    def _is_PentiumMMX(self):
        return self.is_Intel() and self.info[0]['Family']==5 \
               and self.info[0]['Model']==4

    # 判断当前 CPU 是否为 Pentium Pro
    def _is_PentiumPro(self):
        return self.is_Intel() and self.info[0]['Family']==6 \
               and self.info[0]['Model']==1

    # 判断当前 CPU 是否为 Pentium II
    def _is_PentiumII(self):
        return self.is_Intel() and self.info[0]['Family']==6 \
               and self.info[0]['Model'] in [3, 5, 6]

    # 判断当前 CPU 是否为 Pentium III
    def _is_PentiumIII(self):
        return self.is_Intel() and self.info[0]['Family']==6 \
               and self.info[0]['Model'] in [7, 8, 9, 10, 11]

    # 判断当前 CPU 是否为 Pentium IV
    def _is_PentiumIV(self):
        return self.is_Intel() and self.info[0]['Family']==15

    # 判断当前 CPU 是否为 Pentium M
    def _is_PentiumM(self):
        return self.is_Intel() and self.info[0]['Family'] == 6 \
               and self.info[0]['Model'] in [9, 13, 14]

    # 判断当前 CPU 是否为 Core 2
    def _is_Core2(self):
        return self.is_Intel() and self.info[0]['Family'] == 6 \
               and self.info[0]['Model'] in [15, 16, 17]

    # 判断系统是否只有单个 CPU
    def _is_singleCPU(self):
        return len(self.info) == 1

    # 获取系统中 CPU 的数量
    def _getNCPUs(self):
        return len(self.info)

    # 判断当前 CPU 是否支持 MMX 指令集
    def _has_mmx(self):
        if self.is_Intel():
            return (self.info[0]['Family']==5 and self.info[0]['Model']==4) \
                   or (self.info[0]['Family'] in [6, 15])
        elif self.is_AMD():
            return self.info[0]['Family'] in [5, 6, 15]
        else:
            return False

    # 判断当前 CPU 是否支持 SSE 指令集
    def _has_sse(self):
        if self.is_Intel():
            return ((self.info[0]['Family']==6 and
                     self.info[0]['Model'] in [7, 8, 9, 10, 11])
                     or self.info[0]['Family']==15)
        elif self.is_AMD():
            return ((self.info[0]['Family']==6 and
                     self.info[0]['Model'] in [6, 7, 8, 10])
                     or self.info[0]['Family']==15)
        else:
            return False

    # 判断当前 CPU 是否支持 SSE2 指令集
    def _has_sse2(self):
        if self.is_Intel():
            return self.is_Pentium4() or self.is_PentiumM() \
                   or self.is_Core2()
        elif self.is_AMD():
            return self.is_AMD64()
        else:
            return False

    # 判断当前 CPU 是否支持 3DNow 指令集
    def _has_3dnow(self):
        return self.is_AMD() and self.info[0]['Family'] in [5, 6, 15]

    # 判断当前 CPU 是否支持 3DNowExt 指令集
    def _has_3dnowext(self):
        return self.is_AMD() and self.info[0]['Family'] in [6, 15]
# 检查操作系统平台，确定使用哪个 CPU 信息类
if sys.platform.startswith('linux'): # variations: linux2,linux-i386 (any others?)
    # 如果是 Linux 系统，选择 LinuxCPUInfo 类
    cpuinfo = LinuxCPUInfo
elif sys.platform.startswith('irix'):
    # 如果是 IRIX 系统，选择 IRIXCPUInfo 类
    cpuinfo = IRIXCPUInfo
elif sys.platform == 'darwin':
    # 如果是 Darwin（Mac OS X）系统，选择 DarwinCPUInfo 类
    cpuinfo = DarwinCPUInfo
elif sys.platform.startswith('sunos'):
    # 如果是 SunOS 系统，选择 SunOSCPUInfo 类
    cpuinfo = SunOSCPUInfo
elif sys.platform.startswith('win32'):
    # 如果是 Windows 系统，选择 Win32CPUInfo 类
    cpuinfo = Win32CPUInfo
elif sys.platform.startswith('cygwin'):
    # 如果是 Cygwin 系统，选择 LinuxCPUInfo 类（Cygwin 可以通过 LinuxCPUInfo 访问）
    cpuinfo = LinuxCPUInfo
#XXX: other OS's. Eg. use _winreg on Win32. Or os.uname on unices.
else:
    # 如果是其它操作系统，选择 CPUInfoBase 类作为默认
    cpuinfo = CPUInfoBase

# 创建 CPU 信息类的实例
cpu = cpuinfo()

# 下面的代码块不需要注释，因为它们已经被注释掉了，不会执行
```