# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\MemoryFormat.java`

```py
# 定义一个名为 MemoryFormat 的枚举类型，位于 org.pytorch 包下
package org.pytorch;

# 枚举 MemoryFormat 包含以下几种类型
public enum MemoryFormat {
  CONTIGUOUS(1),         # CONTIGUOUS 类型，对应 JNI 中的代码值 1
  CHANNELS_LAST(2),      # CHANNELS_LAST 类型，对应 JNI 中的代码值 2
  CHANNELS_LAST_3D(3),   # CHANNELS_LAST_3D 类型，对应 JNI 中的代码值 3
  ;

  final int jniCode;     # 定义一个 final 的整型变量 jniCode，用于存储 JNI 代码值

  # 枚举类型的构造函数，接收一个 int 类型的参数 jniCode
  MemoryFormat(int jniCode) {
    this.jniCode = jniCode;   # 将参数 jniCode 赋值给成员变量 this.jniCode
  }
}
```