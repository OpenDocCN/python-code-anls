# `basic-computer-games\09_Battle\java\Sea.java`

```
// 跟踪海洋的内容
class Sea {
    // 海洋是一个方形网格，由瓦片组成。它是一个一维数组，这个类将x和y坐标映射到数组索引
    // 每个瓦片要么是空的（索引处的值为0）
    // 要么包含一艘船（索引处的值为船的编号）
    private int tiles[];

    private int size;

    public Sea(int make_size) {
        size = make_size;
        tiles = new int[size*size];
    }

    public int size() { return size; }

    // 这个方法输出海洋的表示，但是按照一种有趣的顺序
    // 这样做的目的是让玩家自己去解决
    public String encodedDump() {
        StringBuilder out = new StringBuilder();
        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y)
                out.append(Integer.toString(get(x, y)));
            out.append('\n');
        }
        return out.toString();
    }

    /* 如果x，y在海里并且是空的，则返回true
     * 如果x，y被占据或超出范围，则返回false
     * 在一个方法中完成这个操作可以更容易地放置船只
     */
    public boolean isEmpty(int x, int y) {
        if ((x<0)||(x>=size)||(y<0)||(y>=size)) return false;
        return (get(x,y) == 0);
    }

    /* 返回船的编号，如果没有船则返回零
     * 与isEmpty(x,y)不同，这些其他方法要求传递的坐标是有效的
     */
    public int get(int x, int y) {
        return tiles[index(x,y)];
    }

    public void set(int x, int y, int value) {
        tiles[index(x, y)] = value;
    }

    // 将坐标映射到数组索引
    private int index(int x, int y) {
        if ((x < 0) || (x >= size))
            throw new ArrayIndexOutOfBoundsException("程序错误：x不能为" + x);
        if ((y < 0) || (y >= size))
            throw new ArrayIndexOutOfBoundsException("程序错误：y不能为" + y);

        return y*size + x;
    }
}
```