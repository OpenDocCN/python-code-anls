# `43_Hammurabi\csharp\ActionResult.cs`

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

需要注释的代码：

```csharp
// Enumerates the different possible outcomes of attempting the various
// actions in the game.
public enum ActionResult
{
    // The action was a success.
    Success,

    // The action could not be completed because the city does not have
    // enough bushels of grain.
    InsufficientStores,

    // ...
        /// The action could not be completed because the city does not have
        /// sufficient acreage.
        /// </summary>
        InsufficientLand,

        /// <summary>
        /// The action could not be completed because the city does not have
        /// sufficient population.
        /// </summary>
        InsufficientPopulation,

        /// <summary>
        /// The requested action offended the city steward.
        /// </summary>
        Offense
    }
}
```

这段代码是一个枚举类型的定义，包括三个枚举值：InsufficientLand、InsufficientPopulation和Offense。每个枚举值都有注释说明其含义。枚举类型通常用于定义一组相关的常量值。
```