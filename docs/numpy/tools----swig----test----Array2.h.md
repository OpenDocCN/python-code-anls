# `.\numpy\tools\swig\test\Array2.h`

```py
#ifndef ARRAY2_H
#define ARRAY2_H

#include "Array1.h"              // 包含 Array1 类的头文件
#include <stdexcept>             // 包含异常处理的标准库
#include <string>                // 包含处理字符串的标准库

class Array2
{
public:

  // 默认构造函数
  Array2();

  // 带大小和数组数据的构造函数
  Array2(int nrows, int ncols, long* data=0);

  // 复制构造函数
  Array2(const Array2 & source);

  // 析构函数
  ~Array2();

  // 赋值运算符重载
  Array2 & operator=(const Array2 & source);

  // 等于运算符重载
  bool operator==(const Array2 & other) const;

  // 获取行数和列数
  int nrows() const;
  int ncols() const;

  // 调整数组大小
  void resize(int nrows, int ncols, long* data);
  void resize(int nrows, int ncols);
  
  // 设置元素访问器
  Array1 & operator[](int i);

  // 获取元素访问器
  const Array1 & operator[](int i) const;

  // 输出为字符串
  std::string asString() const;

  // 获取视图
  void view(int* nrows, int* ncols, long** data) const;

private:
  // 成员变量
  bool _ownData;        // 指示是否拥有数据
  int _nrows;           // 行数
  int _ncols;           // 列数
  long * _buffer;       // 缓冲区指针
  Array1 * _rows;       // 行数组指针

  // 私有方法
  void allocateMemory();     // 分配内存的方法
  void allocateRows();       // 分配行的方法
  void deallocateMemory();   // 释放内存的方法
};

#endif
```