# `46_Hexapawn\csharp\Board.cs`

```
# 导入所需的模块
import zipfile
from io import BytesIO

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
    public Board(params Pawn[] cells)
    {
        _cells = cells;  // 用传入的参数 cells 初始化 _cells 数组
    }

    public Pawn this[int index]
    {
        get => _cells[index - 1];  // 获取 _cells 数组中索引为 index-1 的元素
        set => _cells[index - 1] = value;  // 设置 _cells 数组中索引为 index-1 的元素为 value
    }

    public Board Reflected => new(Cell.AllCells.Select(c => this[c.Reflected]).ToArray());  // 创建一个新的 Board 对象，其中的 _cells 数组由当前对象的每个元素的 Reflected 属性组成

    public IEnumerator<Pawn> GetEnumerator() => _cells.OfType<Pawn>().GetEnumerator();  // 返回一个用于遍历 _cells 数组中 Pawn 类型元素的迭代器

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();  // 实现 IEnumerable 接口的 GetEnumerator 方法，返回 _cells 数组中 Pawn 类型元素的迭代器
    public override string ToString()
    {
        // 创建一个 StringBuilder 对象，用于构建字符串
        var builder = new StringBuilder().AppendLine();
        // 遍历每一行
        for (int row = 0; row < 3; row++)
        {
            // 在每一行前添加空格
            builder.Append("          ");
            // 遍历每一列
            for (int col = 0; col < 3; col++)
            {
                // 将每个单元格的值添加到字符串构建器中
                builder.Append(_cells[row * 3 + col]);
            }
            // 在每一行结束时添加换行符
            builder.AppendLine();
        }
        // 返回构建好的字符串
        return builder.ToString();
    }

    public bool Equals(Board other) => other?.Zip(this).All(x => x.First == x.Second) ?? false;
    // 检查当前对象是否与另一个对象相等
    public override bool Equals(object obj) => Equals(obj as Board);
    # 重写 GetHashCode 方法
    public override int GetHashCode()
    {
        # 初始化哈希值
        var hash = 19;

        # 遍历_cells数组的前9个元素，计算哈希值
        for (int i = 0; i < 9; i++)
        {
            hash = hash * 53 + _cells[i].GetHashCode();
        }

        # 返回计算得到的哈希值
        return hash;
    }
```
在这段代码中，我们重写了 GetHashCode 方法，该方法用于计算对象的哈希值。首先初始化了一个哈希值，然后遍历了_cells数组的前9个元素，对每个元素的哈希值进行计算并与之前的哈希值进行组合，最终返回计算得到的哈希值。
```