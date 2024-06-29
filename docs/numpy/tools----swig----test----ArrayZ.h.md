# `.\numpy\tools\swig\test\ArrayZ.h`

```
#ifndef ARRAYZ_H
#define ARRAYZ_H

#include <stdexcept>
#include <string>
#include <complex>

class ArrayZ
{
public:

  // 默认/长度/数组构造函数
  ArrayZ(int length = 0, std::complex<double>* data = 0);

  // 复制构造函数
  ArrayZ(const ArrayZ & source);

  // 析构函数
  ~ArrayZ();

  // 赋值运算符重载
  ArrayZ & operator=(const ArrayZ & source);

  // 等于运算符重载
  bool operator==(const ArrayZ & other) const;

  // 长度访问器
  int length() const;

  // 调整数组大小
  void resize(int length, std::complex<double>* data = 0);

  // 设置元素访问器
  std::complex<double> & operator[](int i);

  // 获取元素访问器
  const std::complex<double> & operator[](int i) const;

  // 字符串输出
  std::string asString() const;

  // 获取视图
  void view(std::complex<double>** data, int* length) const;

private:
  // 成员变量
  bool _ownData;                         // 是否拥有数据的标志
  int _length;                           // 数组长度
  std::complex<double> * _buffer;        // 数据缓冲区指针

  // 私有方法
  void allocateMemory();                 // 分配内存方法
  void deallocateMemory();               // 释放内存方法
};

#endif
```