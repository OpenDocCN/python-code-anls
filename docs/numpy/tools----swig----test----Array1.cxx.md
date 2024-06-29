# `.\numpy\tools\swig\test\Array1.cxx`

```
// 引入自定义头文件 Array1.h
#include "Array1.h"
// 引入标准输入输出流库
#include <iostream>
// 引入字符串流库
#include <sstream>

// Array1 类的构造函数，支持默认、长度、数组输入
Array1::Array1(int length, long* data) :
  _ownData(false), _length(0), _buffer(0)
{
  // 调整数组大小并分配内存
  resize(length, data);
}

// Array1 类的拷贝构造函数
Array1::Array1(const Array1 & source) :
  _length(source._length)
{
  // 分配内存并复制源对象的数据
  allocateMemory();
  *this = source;
}

// Array1 类的析构函数，释放内存
Array1::~Array1()
{
  // 释放对象所持有的内存
  deallocateMemory();
}

// Array1 类的赋值运算符重载
Array1 & Array1::operator=(const Array1 & source)
{
  // 比较长度并选择较小的长度进行赋值
  int len = _length < source._length ? _length : source._length;
  for (int i=0;  i < len; ++i)
  {
    (*this)[i] = source[i];
  }
  return *this;
}

// Array1 类的相等比较运算符重载
bool Array1::operator==(const Array1 & other) const
{
  // 比较数组长度及每个元素是否相等
  if (_length != other._length) return false;
  for (int i=0; i < _length; ++i)
  {
    if ((*this)[i] != other[i]) return false;
  }
  return true;
}

// 获取数组长度的访问器
int Array1::length() const
{
  return _length;
}

// 调整数组大小的方法
void Array1::resize(int length, long* data)
{
  // 检查长度是否小于零，抛出异常
  if (length < 0) throw std::invalid_argument("Array1 length less than 0");
  // 若长度与当前长度相同，则直接返回
  if (length == _length) return;
  // 释放当前内存，并重新分配新的内存
  deallocateMemory();
  _length = length;
  // 如果没有指定数据，则分配新内存
  if (!data)
  {
    allocateMemory();
  }
  else
  {
    // 否则使用提供的数据并设置标志
    _ownData = false;
    _buffer  = data;
  }
}

// 设置数组元素的访问器
long & Array1::operator[](int i)
{
  // 检查索引是否越界，抛出异常
  if (i < 0 || i >= _length) throw std::out_of_range("Array1 index out of range");
  return _buffer[i];
}

// 获取数组元素的访问器（常量版本）
const long & Array1::operator[](int i) const
{
  // 检查索引是否越界，抛出异常
  if (i < 0 || i >= _length) throw std::out_of_range("Array1 index out of range");
  return _buffer[i];
}

// 以字符串形式输出数组
std::string Array1::asString() const
{
  // 使用字符串流生成数组的字符串表示
  std::stringstream result;
  result << "[";
  for (int i=0; i < _length; ++i)
  {
    result << " " << _buffer[i];
    if (i < _length-1) result << ",";
  }
  result << " ]";
  return result.str();
}

// 获取数组视图的方法
void Array1::view(long** data, int* length) const
{
  // 返回数组缓冲区及其长度
  *data   = _buffer;
  *length = _length;
}

// Array1 类的私有方法，分配内存
void Array1::allocateMemory()
{
  // 如果长度为零，则设置标志并置空缓冲区
  if (_length == 0)
  {
    _ownData = false;
    _buffer  = 0;
  }
  else
  {
    // 否则分配新内存，并设置标志
    _ownData = true;
    _buffer = new long[_length];
  }
}

// Array1 类的私有方法，释放内存
void Array1::deallocateMemory()
{
  // 如果对象持有数据并且缓冲区不为空，则释放内存
  if (_ownData && _length && _buffer)
  {
    delete [] _buffer;
  }
  // 重置标志及长度，置空缓冲区
  _ownData = false;
  _length  = 0;
  _buffer  = 0;
}
```