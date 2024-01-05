# `09_Battle\java\Sea.java`

```
// 跟踪海洋的内容
class Sea {
    // 海洋是一个方形网格的瓦片。它是一个一维数组，这个类将x和y坐标映射到数组索引
    // 每个瓦片要么是空的（索引处的瓦片值为0）
    // 要么包含一艘船（索引处的瓦片值为船的编号）
    private int tiles[];

    private int size;

    public Sea(int make_size) {
        size = make_size;
        tiles = new int[size*size];
    }

    public int size() { return size; }

    // 这个方法输出海洋的表示，但是按照一种有趣的顺序
    // 这样做的想法是让玩家来解决它
    public String encodedDump() {
        // 创建一个 StringBuilder 对象，用于构建字符串
        StringBuilder out = new StringBuilder();
        // 遍历二维数组，将每个元素转换为字符串并添加到 StringBuilder 对象中
        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y)
                out.append(Integer.toString(get(x, y)));
            // 在每行末尾添加换行符
            out.append('\n');
        }
        // 将 StringBuilder 对象转换为字符串并返回
        return out.toString();
    }

    /* 如果坐标 (x, y) 在海域中且为空，则返回 true
     * 如果坐标 (x, y) 被占据或超出范围，则返回 false
     * 在一个方法中完成这些操作可以更轻松地放置船只
     */
    public boolean isEmpty(int x, int y) {
        // 如果坐标超出范围，则返回 false
        if ((x<0)||(x>=size)||(y<0)||(y>=size)) return false;
        // 如果坐标处为空，则返回 true
        return (get(x,y) == 0);
    }

    /* 返回船只编号，如果没有船只则返回零
     * 与 isEmpty(x, y) 不同，这些方法要求
     *```
    /**
     * 根据传入的坐标获取对应位置的值
     */
    public int get(int x, int y) {
        return tiles[index(x,y)];
    }

    /**
     * 根据传入的坐标设置对应位置的值
     */
    public void set(int x, int y, int value) {
        tiles[index(x, y)] = value;
    }

    // 将坐标映射到数组索引
    private int index(int x, int y) {
        if ((x < 0) || (x >= size))
            throw new ArrayIndexOutOfBoundsException("Program error: x cannot be " + x);
        if ((y < 0) || (y >= size))
            throw new ArrayIndexOutOfBoundsException("Program error: y cannot be " + y);

        return y*size + x;
    }
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```