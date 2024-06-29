# `.\numpy\tools\swig\test\ArrayZ.cxx`

```
// 包含自定义头文件 ArrayZ.h 和标准输入输出流头文件
#include "ArrayZ.h"
#include <iostream>
#include <sstream>

// 默认/长度/数组构造函数
ArrayZ::ArrayZ(int length, std::complex<double>* data) :
  _ownData(false), _length(0), _buffer(0)
{
  // 调整数组大小并分配内存
  resize(length, data);
}

// 拷贝构造函数
ArrayZ::ArrayZ(const ArrayZ & source) :
  _length(source._length)
{
  // 分配内存并复制源对象数据
  allocateMemory();
  *this = source;
}

// 析构函数
ArrayZ::~ArrayZ()
{
  // 释放对象的内存
  deallocateMemory();
}

// 赋值运算符重载
ArrayZ & ArrayZ::operator=(const ArrayZ & source)
{
  // 比较并复制长度较小的数据
  int len = _length < source._length ? _length : source._length;
  for (int i=0;  i < len; ++i)
  {
    (*this)[i] = source[i];
  }
  return *this;
}

// 等号运算符重载
bool ArrayZ::operator==(const ArrayZ & other) const
{
  // 比较数组长度及每个元素是否相等
  if (_length != other._length) return false;
  for (int i=0; i < _length; ++i)
  {
    if ((*this)[i] != other[i]) return false;
  }
  return true;
}

// 返回数组长度的访问器
int ArrayZ::length() const
{
  return _length;
}

// 调整数组大小
void ArrayZ::resize(int length, std::complex<double>* data)
{
  // 检查长度是否合法
  if (length < 0) throw std::invalid_argument("ArrayZ length less than 0");
  // 若长度不变则直接返回
  if (length == _length) return;
  // 释放当前内存并重新分配
  deallocateMemory();
  _length = length;
  if (!data)
  {
    // 分配新的内存
    allocateMemory();
  }
  else
  {
    // 使用传入的数据作为数组缓冲区
    _ownData = false;
    _buffer  = data;
  }
}

// 设置元素访问器
std::complex<double> & ArrayZ::operator[](int i)
{
  // 检查索引是否有效
  if (i < 0 || i >= _length) throw std::out_of_range("ArrayZ index out of range");
  return _buffer[i];
}

// 获取元素访问器（常量版本）
const std::complex<double> & ArrayZ::operator[](int i) const
{
  // 检查索引是否有效
  if (i < 0 || i >= _length) throw std::out_of_range("ArrayZ index out of range");
  return _buffer[i];
}

// 生成数组的字符串表示
std::string ArrayZ::asString() const
{
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

// 获取数组视图
void ArrayZ::view(std::complex<double>** data, int* length) const
{
  // 返回数组缓冲区及其长度
  *data   = _buffer;
  *length = _length;
}

// 私有方法：分配内存
void ArrayZ::allocateMemory()
{
  if (_length == 0)
  {
    _ownData = false;
    _buffer  = 0;
  }
  else
  {
    // 分配新的数组内存
    _ownData = true;
    _buffer = new std::complex<double>[_length];
  }
}

// 私有方法：释放内存
void ArrayZ::deallocateMemory()
{
  // 如果拥有数据并且数组长度大于0，则释放内存
  if (_ownData && _length && _buffer)
  {
    delete [] _buffer;
  }
  _ownData = false;
  _length  = 0;
  _buffer  = 0;
}
```