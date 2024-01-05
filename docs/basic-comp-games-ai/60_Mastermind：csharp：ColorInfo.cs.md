# `d:/src/tocomm/basic-computer-games\60_Mastermind\csharp\ColorInfo.cs`

```
// 声明一个命名空间 Game
namespace Game
{
    /// <summary>
    /// 存储有关颜色的信息。
    /// </summary>
    // 声明一个记录类型 ColorInfo
    public record ColorInfo
    {
        /// <summary>
        /// 获取表示颜色的单个字符。
        /// </summary>
        // 声明一个属性 ShortName，用于获取颜色的简称
        public char ShortName { get; init; }

        /// <summary>
        /// 获取颜色的全名。
        /// </summary>
        // 声明一个属性 LongName，用于获取颜色的全名，并初始化为空字符串
        public string LongName { get; init; } = String.Empty;
    }
}
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