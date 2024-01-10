# `basic-computer-games\09_Battle\java\Ship.java`

```
# 导入需要的 Java 类
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.function.Predicate;

# 定义船只类，包括位置和被击中的情况
class Ship {
    # 这是船只可能的四个方向
    public static final int ORIENT_E=0;   // 从起始位置向东
    public static final int ORIENT_SE=1;  // 从起始位置向东南
    public static final int ORIENT_S=2;   // 从起始位置向南
    public static final int ORIENT_SW=3;  // 从起始位置向西南

    private int id;                   // 船只编号
    private int size;                 // 占据的格子数
    private boolean placed;           // 船只是否已经在海里
    private boolean sunk;             // 船只是否已经沉没
    private ArrayList<Boolean> hits;  // 船只的哪些格子被击中

    private int startX;               // 起始位置的坐标
    private int startY;
    private int orientX;              // 每个占据的格子到下一个格子的 x 和 y 增量
    private int orientY;

    public Ship(int i, int sz) {
        id = i; size = sz;
        sunk = false; placed = false;
        hits = new ArrayList<>(Collections.nCopies(size, false));
    }

    /** @returns 船只编号 */
    public int id() { return id; }
    /** @returns 船只大小 */
    public int size() { return size; }

    /* 记录船只在给定坐标被击中 */
    // 当击中时，需要计算从船的起始位置到击中位置的瓦片数
    // 可以通过起始 X 坐标和当前 X 坐标的差值来计算
    // 除非船是南北方向的，那么就使用 Y 坐标
    public void hit(int x, int y) {
        int offset;
        if (orientX != 0) {
            offset = (x - startX) / orientX;
        } else {
            offset = (y - startY) / orientY;
        }
        hits.set(offset, true);

        // 如果船的每个瓦片都被击中，那么船就被击沉了
        sunk = hits.stream().allMatch(Predicate.isEqual(true));
    }

    // 返回船是否被击沉
    public boolean isSunk() { return sunk; }

    // 检查给定坐标是否已经击中船
    public boolean wasHit(int x, int y) {
        int offset;
        if (orientX != 0) {
            offset = (x - startX) / orientX;
        } else {
            offset = (y - startY) / orientY;
        }
        return hits.get(offset);
    };

    // 在海洋中放置船
    // 选择一个随机的起始位置和一个随机的方向
    // 如果不适合，就不断选择不同的位置和方向
    public void placeRandom(Sea s) {
        Random random = new Random();
        for (int tries = 0 ; tries < 1000 ; ++tries) {
            int x = random.nextInt(s.size());
            int y = random.nextInt(s.size());
            int orient = random.nextInt(4);

            if (place(s, x, y, orient)) return;
        }

        throw new RuntimeException("Could not place any more ships");
    }

    // 尝试将船放入海洋中，从给定位置和给定方向开始
    // 这是程序中最复杂的部分
    // 它将从提供的位置开始，并尝试在请求的方向上占据瓦片
    // 如果不适合，要么是因为到达了海洋的边缘，要么是因为已经有船在那里，它将尝试扩展船只
    // 在相反的方向上扩展船只，如果不可能，则失败。

    // 检查已经占据“from”坐标的船只是否也可以占据“to”坐标。
    // 它们必须在海域内，为空，并且不会导致船只相互交叉
    private boolean extendShip(Sea s, int fromX, int fromY, int toX, int toY) {
        if (!s.isEmpty(toX, toY)) return false;                  // 没有空间
        if ((fromX == toX)||(fromY == toY)) return true;         // 水平或垂直

        // 我们可以在不发生碰撞的情况下扩展船只，但是我们是在对角线上
        // 两艘船只不应该在对角线上相互交叉。

        // 检查这里会交叉的两个方块 - 如果任何一个是空的，那么我们是安全的
        // 如果它们都包含不同的船只，那么我们是安全的
        // 但是如果它们都包含相同的船只，那么我们就是在交叉！
        int corner1 = s.get(fromX, toY);
        int corner2 = s.get(toX, fromY);
        if ((corner1 == 0) || (corner1 != corner2)) return true;
        return false;
    }
# 闭合前面的函数定义
```