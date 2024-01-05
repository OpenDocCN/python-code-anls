# `d:/src/tocomm/basic-computer-games\71_Poker\csharp\Cards\Rank.cs`

```
    public static Rank King = new(13, "King");
    public static Rank Ace = new(14, "Ace");

    private readonly int _value;
    private readonly string _name;

    public Rank(int value, string name = null)
    {
        _value = value;
        _name = name ?? value.ToString();
    }

    public int CompareTo(Rank other)
    {
        return _value.CompareTo(other._value);
    }
}
```

注释：

1. `namespace Poker.Cards;` - 声明了代码所在的命名空间为Poker.Cards。

2. `internal struct Rank : IComparable<Rank>` - 声明了一个内部的结构体Rank，并实现了IComparable接口，用于比较Rank对象。

3. `public static IEnumerable<Rank> Ranks => new[] { ... }` - 声明了一个公共的静态属性Ranks，返回一个包含所有Rank对象的可枚举集合。

4. `public static Rank Two = new(2);` - 声明了一个公共的静态属性Two，表示牌面为2的Rank对象。

5. `private readonly int _value;` - 声明了一个私有的只读字段_value，用于存储Rank对象的值。

6. `private readonly string _name;` - 声明了一个私有的只读字段_name，用于存储Rank对象的名称。

7. `public Rank(int value, string name = null)` - 声明了一个公共的构造函数，用于初始化Rank对象的值和名称。

8. `public int CompareTo(Rank other)` - 声明了一个公共的方法CompareTo，用于比较当前Rank对象和另一个Rank对象的值。
    # 创建一个公共的 Rank 对象 King，值为 13，名称为 "King"
    King = Rank(13, "King")
    # 创建一个公共的 Rank 对象 Ace，值为 14，名称为 "Ace"
    Ace = Rank(14, "Ace")

    # 定义私有的 _value 和 _name 属性
    # _value 用于存储牌面值，_name 用于存储牌面名称
    # Rank 对象的构造函数，接受一个值和一个可选的名称参数
    def __init__(self, value, name=None):
        self._value = value
        # 如果没有传入名称参数，则使用默认名称为值的字符串
        self._name = name if name is not None else f" {value} "

    # 重写 ToString 方法，返回牌面名称
    def __str__(self):
        return self._name

    # 实现 CompareTo 方法，用于比较两个 Rank 对象的大小
    def __cmp__(self, other):
        return self._value - other._value

    # 实现小于运算符重载，用于比较两个 Rank 对象的大小
    def __lt__(self, other):
        return self._value < other._value

    # 实现大于运算符重载，用于比较两个 Rank 对象的大小
    def __gt__(self, other):
        return self._value > other._value

    # 实现等于运算符重载，用于比较两个 Rank 对象的大小
    def __eq__(self, other):
        return self._value == other._value

    # 实现不等于运算符重载，用于比较两个 Rank 对象的大小
    def __ne__(self, other):
        return self._value != other._value
# 定义一个名为 read_zip 的函数，接受一个参数 fname，用于根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
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