# `.\numpy\tools\swig\test\Array1.h`

```py
#ifndef ARRAY1_H
#define ARRAY1_H

#include <stdexcept>
#include <string>

class Array1
{
public:
  // Default/length/array constructor
  // 默认构造函数，可以指定长度和数据
  Array1(int length = 0, long* data = 0);

  // Copy constructor
  // 拷贝构造函数
  Array1(const Array1 & source);

  // Destructor
  // 析构函数
  ~Array1();

  // Assignment operator
  // 赋值运算符重载
  Array1 & operator=(const Array1 & source);

  // Equals operator
  // 等号运算符重载，用于比较两个对象是否相等
  bool operator==(const Array1 & other) const;

  // Length accessor
  // 获取数组长度
  int length() const;

  // Resize array
  // 调整数组大小
  void resize(int length, long* data = 0);

  // Set item accessor
  // 设置数组元素
  long & operator[](int i);

  // Get item accessor
  // 获取数组元素（常量版本）
  const long & operator[](int i) const;

  // String output
  // 返回数组的字符串表示
  std::string asString() const;

  // Get view
  // 获取数组数据和长度的视图
  void view(long** data, int* length) const;

private:
  // Members
  // 成员变量
  bool _ownData;  // 是否拥有数据
  int _length;    // 数组长度
  long * _buffer; // 数据缓冲区

  // Methods
  // 私有方法
  void allocateMemory();    // 分配内存
  void deallocateMemory();  // 释放内存
};

#endif
```