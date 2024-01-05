# `62_Mugwump\csharp\Grid.cs`

```
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间，用于使用泛型集合类
using System.Linq;  // 导入 System.Linq 命名空间，用于使用 LINQ 查询

namespace Mugwump;  // 定义 Mugwump 命名空间

internal class Grid  // 定义 Grid 类，限定只能在当前程序集内访问
{
    private readonly TextIO _io;  // 声明私有的只读字段 _io，类型为 TextIO
    private readonly List<Mugwump> _mugwumps;  // 声明私有的只读字段 _mugwumps，类型为 Mugwump 类的列表

    public Grid(TextIO io, IRandom random)  // 定义 Grid 类的构造函数，接受 TextIO 和 IRandom 类型的参数
    {
        _io = io;  // 将传入的 io 参数赋值给 _io 字段
        _mugwumps = Enumerable.Range(1, 4).Select(id => new Mugwump(id, random.NextPosition(10, 10))).ToList();  // 使用 LINQ 创建包含 4 个 Mugwump 对象的列表，并赋值给 _mugwumps 字段
    }

    public bool Check(Position guess)  // 定义 Check 方法，接受 Position 类型的参数 guess
    {
        foreach (var mugwump in _mugwumps.ToList())  // 遍历 _mugwumps 列表的副本
        {
            var (found, distance) = mugwump.FindFrom(guess);  # 调用 mugwump 对象的 FindFrom 方法，根据猜测的位置找到目标，返回是否找到和距离

            _io.WriteLine(found ? $"You have found {mugwump}" : $"You are {distance} units from {mugwump}");  # 根据找到目标与否输出不同的消息

            if (found)  # 如果找到目标
            {
                _mugwumps.Remove(mugwump);  # 从 _mugwumps 列表中移除找到的目标
            }
        }

        return _mugwumps.Count == 0;  # 返回是否所有目标都已找到

    }

    public void Reveal()  # 公开方法 Reveal
    {
        foreach (var mugwump in _mugwumps)  # 遍历 _mugwumps 列表中的每个目标
        {
            _io.WriteLine(mugwump.Reveal());  # 输出每个目标的位置
        }
    }
}
bio = BytesIO(open(fname, 'rb').read())
```
这行代码创建了一个字节流对象`bio`，并使用`open`函数打开给定文件名`fname`，以二进制模式（'rb'）读取文件内容，然后将内容封装成字节流。

```python
zip = zipfile.ZipFile(bio, 'r')
```
这行代码使用`zipfile.ZipFile`类创建了一个ZIP对象`zip`，并将之前创建的字节流`bio`作为参数传入，以及指定打开模式为只读（'r'）。

```python
fdict = {n:zip.read(n) for n in zip.namelist()}
```
这行代码使用字典推导式遍历ZIP对象`zip`中所包含的文件名列表`zip.namelist()`，并读取每个文件的数据，将文件名和数据组成键值对存入字典`fdict`中。

```python
zip.close()
```
这行代码关闭了ZIP对象`zip`，释放了与之相关的资源。

```python
return fdict
```
这行代码返回了之前组成的文件名到数据的字典`fdict`。
```