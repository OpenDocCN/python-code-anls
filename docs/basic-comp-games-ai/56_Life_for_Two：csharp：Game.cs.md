# `56_Life_for_Two\csharp\Game.cs`

```
internal class Game
{
    private readonly IReadWrite _io;  # 声明私有变量 _io，类型为 IReadWrite 接口

    public Game(IReadWrite io)  # 构造函数，接受一个 IReadWrite 类型的参数 io
    {
        _io = io;  # 将传入的 io 参数赋值给私有变量 _io
    }

    public void Play()  # Play 方法
    {
        _io.Write(Streams.Title);  # 调用 _io 对象的 Write 方法，输出 Streams.Title 的内容

        var life = new Life(_io);  # 创建 Life 类的实例，传入 _io 对象作为参数

        _io.Write(life.FirstGeneration);  # 调用 _io 对象的 Write 方法，输出 life.FirstGeneration 的内容

        foreach (var generation in life)  # 遍历 life 对象
        {
            _io.WriteLine();  # 调用 _io 对象的 WriteLine 方法，输出空行
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名列表，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```