# `KubiScan\engine\capabilities\capabilities.py`

```
# 创建一个包含系统权限名称和对应数字值的字典
caps_list = {
  "CHOWN": 1,  # 更改文件所有者
  "DAC_OVERRIDE": 2,  # 忽略文件的 DAC 权限
  "DAC_READ_SEARCH": 3,  # 可读取和搜索文件
  "FOWNER": 4,  # 忽略目录的 DAC 权限
  "FSETID": 5,  # 文件的 setuid 和 setgid 位
  "KILL": 6,  # 发送信号给任何进程
  "SETGID": 7,  # 设置组 ID
  "SETUID": 8,  # 设置用户 ID
  "SETPCAP": 9,  # 修改进程的能力
  "LINUX_IMMUTABLE": 10,  # 设置文件的不可变位
  "NET_BIND_SERVICE": 11,  # 绑定到小于 1024 的端口
  "NET_BROADCAST": 12,  # 发送广播信息
  "NET_ADMIN": 13,  # 进行网络配置
  "NET_RAW": 14,  # 使用原始套接字
  "IPC_LOCK": 15,  # 锁定内存
  "IPC_OWNER": 16,  # 拥有 IPC 所有权
  "SYS_MODULE": 17,  # 加载内核模块
  "SYS_RAWIO": 18,  # 执行原始 I/O 操作
# 定义一系列系统调用的常量，每个常量对应一个系统调用的编号
"SYS_CHROOT": 19,  # 切换根目录
"SYS_PTRACE": 20,  # 追踪进程
"SYS_PACCT": 21,   # 进程记账
"SYS_ADMIN": 22,   # 管理系统
"SYS_BOOT": 23,    # 重启系统
"SYS_NICE": 24,    # 修改进程优先级
"SYS_RESOURCE": 25,  # 控制资源
"SYS_TIME": 26,    # 修改系统时间
"SYS_TTY_CONFIG": 27,  # 修改终端设备配置
"MKNOD": 28,       # 创建设备文件
"LEASE": 29,       # 文件租约
"AUDIT_WRITE": 30,  # 写入审计记录
"AUDIT_CONTROL": 31,  # 控制审计功能
"SETFCAP": 32,     # 设置文件能力
"MAC_OVERRIDE": 33,  # 覆盖 MAC 安全机制
"MAC_ADMIN": 34,   # 管理 MAC 安全机制
"SYSLOG": 35,      # 写入系统日志
"WAKE_ALARM": 36,  # 设置唤醒闹钟
"BLOCK_SUSPEND": 37,  # 阻止系统挂起
"AUDIT_READ": 38   # 读取审计记录
# 定义一个默认的权限字典，包含了各种权限名称和对应的数值
default_caps = {
  "CAP_CHOWN": 1,
  "DAC_OVERRIDE": 2,
  "FOWNER": 4,
  "FSETID": 5,
  "KILL": 6,
  "SETGID": 7,
  "SETUID": 8,
  "SETPCAP": 9,
  "NET_BIND_SERVICE": 11,
  "NET_RAW": 14,
  "SYS_CHROOT": 19,
  "MKNOD": 28,
  "AUDIT_WRITE": 30,
  "SETFCAP": 32,
  "AUDIT_READ": 38
}
# 定义一个包含危险权限和对应数值的字典
dangerous_caps = {
  "DAC_READ_SEARCH": 3,  # 文件读取和搜索权限
  "LINUX_IMMUTABLE": 10,  # Linux不可变权限
  "NET_BROADCAST": 12,  # 网络广播权限
  "NET_ADMIN": 13,  # 网络管理权限
  "IPC_LOCK": 15,  # 进程间通信锁定权限
  "IPC_OWNER": 16,  # 进程间通信所有者权限
  "SYS_MODULE": 17,  # 内核模块加载和卸载权限
  "SYS_RAWIO": 18,  # 原始IO权限
  "SYS_PTRACE": 20,  # 进程跟踪权限
  "SYS_BOOT": 23,  # 系统引导权限
  "SYS_PACCT": 21,  # 进程账户权限
  "SYS_ADMIN": 22,  # 系统管理权限
  "SYS_NICE": 24,  # 系统优先级权限
  "SYS_RESOURCE": 25,  # 系统资源权限
  "SYS_TIME": 26,  # 系统时间权限
  "SYS_TTY_CONFIG": 27,  # 终端设备配置权限
  "LEASE": 29,  # 文件租约权限
  "AUDIT_CONTROL": 31,  # 审计控制权限
  "MAC_OVERRIDE": 33,  # MAC覆盖权限
}
# 定义一个字典，包含不同的系统权限和对应的数值
{
  "MAC_ADMIN": 34,  # MAC 管理员权限
  "SYSLOG": 35,     # 系统日志权限
  "WAKE_ALARM": 36, # 唤醒闹钟权限
  "BLOCK_SUSPEND": 37  # 阻止挂起权限
}

# 下面的代码是注释，没有实际的功能，可以忽略
#indexes = get_indexes_with_one(0x10)  # 获取数值为 0x10 的索引
#indexes = get_indexes_with_one(0x3fffffffff)  # 获取数值为 0x3fffffffff 的索引
#print_decoded_capabilities(indexes)  # 打印解码后的权限信息
```