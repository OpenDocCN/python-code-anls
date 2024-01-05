# `d:/src/tocomm/basic-computer-games\29_Craps\csharp\Craps\Dice.cs`

```
# 创建一个名为Dice的类
class Dice:
    # 初始化一个Random对象作为rand，并设置sides为只读属性
    def __init__(self):
        # 初始化一个Random对象作为rand
        self.rand = Random()
        # 设置sides属性为6
        self.sides = 6

    # 初始化一个带参数的构造函数，接受sides作为参数
    def __init__(self, sides):
        # 设置sides属性为传入的参数值
        self.sides = sides
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```
```csharp
public int Roll() => rand.Next(1, sides + 1);
```

注释：
- 这是一个 C# 语言的方法定义，用于模拟掷骰子的行为。
- 方法名为 Roll，返回一个整数值。
- 使用了箭头函数的语法，等同于传统的方法定义方式。
- 方法内部调用了 rand.Next 方法，生成一个介于 1 和 sides+1 之间的随机整数。
```