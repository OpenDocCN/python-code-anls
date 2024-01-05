# `d:/src/tocomm/basic-computer-games\17_Bullfight\csharp\Reward.cs`

```
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

在这段代码中，我们定义了一个函数read_zip，它接受一个ZIP文件名作为参数，并返回一个包含文件名到数据的字典。函数内部的注释解释了每个语句的作用。bio变量封装了ZIP文件的二进制内容，zip变量使用这个字节流创建了一个ZIP对象，fdict变量则包含了ZIP文件中的文件名和对应的数据。最后，我们关闭了ZIP对象并返回结果字典。

```csharp
namespace Game
{
    /// <summary>
    /// Enumerates the different things the player can be awarded.
    /// </summary>
    public enum Reward
    {
        Nothing,
        OneEar,
        TwoEars,
        CarriedFromRing
    }
}
```

在这段C#代码中，我们定义了一个枚举类型Reward，它列举了玩家可以获得的不同奖励。注释解释了这个枚举类型的作用。
```