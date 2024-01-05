# `d:/src/tocomm/basic-computer-games\17_Bullfight\csharp\Action.cs`

```
/// <summary>
/// Enumerates the different actions that the player can take on each round
/// of the fight.
/// </summary>
public enum Action
{
    /// <summary>
    /// Dodge the bull.
    /// </summary>
    Dodge,

    /// <summary>
    /// Kill the bull.
    /// </summary>
    Kill,

    /// <summary>
    /// Freeze in place and don't do anything.
    /// </summary>
    Freeze
}
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```