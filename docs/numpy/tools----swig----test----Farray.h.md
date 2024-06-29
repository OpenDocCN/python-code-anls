# `.\numpy\tools\swig\test\Farray.h`

```py
#ifndef FARRAY_H
#define FARRAY_H

#include <stdexcept>
#include <string>

class Farray
{
public:
  // Size constructor
  Farray(int nrows, int ncols);

  // Copy constructor
  Farray(const Farray & source);

  // Destructor
  ~Farray();

  // Assignment operator
  Farray & operator=(const Farray & source);

  // Equals operator
  bool operator==(const Farray & other) const;

  // Length accessors
  int nrows() const;
  int ncols() const;

  // Set item accessor
  long & operator()(int i, int j);

  // Get item accessor
  const long & operator()(int i, int j) const;

  // String output
  std::string asString() const;

  // Get view
  void view(int* nrows, int* ncols, long** data) const;

private:
  // Members
  int _nrows;       // 行数
  int _ncols;       // 列数
  long * _buffer;   // 数据存储缓冲区

  // Default constructor: not implemented
  Farray();

  // Methods
  void allocateMemory();   // 分配内存方法
  int  offset(int i, int j) const;   // 计算索引偏移量的方法
};

#endif


这些注释提供了每个类成员和方法的简要解释，帮助理解其用途和功能。
```