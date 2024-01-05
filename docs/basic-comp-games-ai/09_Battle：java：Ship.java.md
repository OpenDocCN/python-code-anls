# `09_Battle\java\Ship.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Collections;  // 导入 Collections 类
import java.util.Comparator;   // 导入 Comparator 类
import java.util.Random;       // 导入 Random 类
import java.util.function.Predicate;  // 导入 Predicate 接口

/** A single ship, with its position and where it has been hit */
class Ship {
    // These are the four directions that ships can be in
    public static final int ORIENT_E=0;   // 定义船只可能的方向：向东
    public static final int ORIENT_SE=1;  // 定义船只可能的方向：东南
    public static final int ORIENT_S=2;   // 定义船只可能的方向：向南
    public static final int ORIENT_SW=3;  // 定义船只可能的方向：西南

    private int id;                   // 船只编号
    private int size;                 // 船只占据的格子数
    private boolean placed;           // 船只是否已经放置在海上
    private boolean sunk;             // 船只是否已经沉没
    private ArrayList<Boolean> hits;  // 船只被击中的格子
    private int startX;               // starting position coordinates - 定义起始位置的 x 坐标
    private int startY;               // starting position coordinates - 定义起始位置的 y 坐标
    private int orientX;              // x and y deltas from each tile occupied to the next - 从每个占据的瓦片到下一个瓦片的 x 和 y 增量
    private int orientY;              // x and y deltas from each tile occupied to the next - 从每个占据的瓦片到下一个瓦片的 x 和 y 增量

    public Ship(int i, int sz) {      // 构造函数，初始化船只的编号和大小
        id = i; size = sz;            // 设置船只的编号和大小
        sunk = false; placed = false; // 初始化船只的状态为未沉没和未放置
        hits = new ArrayList<>(Collections.nCopies(size, false)); // 创建一个大小为船只大小的列表，用于记录船只被击中的情况
    }

    /** @returns the ship number */    // 返回船只的编号
    public int id() { return id; }      // 返回船只的编号
    /** @returns the ship size */       // 返回船只的大小
    public int size() { return size; }  // 返回船只的大小

    /* record the ship as having been hit at the given coordinates */  // 记录船只在给定坐标处被击中
    public void hit(int x, int y) {    // 记录船只在给定坐标处被击中
        // need to work out how many tiles from the ship's starting position the hit is at
        // that can be worked out from the difference between the starting X coord and this one
        // 需要计算从船只的起始位置到击中位置有多少个瓦片
        // 可以通过起始 X 坐标和当前 X 坐标的差值来计算
        // 除非船只朝南北方向航行，否则使用 Y 坐标
        int offset;
        if (orientX != 0) {
            offset = (x - startX) / orientX;
        } else {
            offset = (y - startY) / orientY;
        }
        hits.set(offset, true); // 在 hits 集合中设置指定偏移量的元素为 true，表示船只在该位置被击中

        // 如果船只的每个方格都被击中，那么船只已经沉没
        sunk = hits.stream().allMatch(Predicate.isEqual(true)); // 使用流操作检查 hits 集合中的所有元素是否都为 true，如果是则船只已经沉没
    }

    public boolean isSunk() { return sunk; } // 返回船只是否已经沉没的状态

    // 检查船只是否在给定坐标处已经被击中
    public boolean wasHit(int x, int y) {
        int offset;
        if (orientX != 0) {
            offset = (x - startX) / orientX; // 根据船只的方向计算击中位置的偏移量
        } else {
            offset = (y - startY) / orientY;  // 计算偏移量，根据起始位置和方向
        }
        return hits.get(offset);  // 返回偏移量对应的结果
    };

    // Place the ship in the sea.
    // choose a random starting position, and a random direction
    // if that doesn't fit, keep picking different positions and directions
    public void placeRandom(Sea s) {
        Random random = new Random();  // 创建一个随机数生成器
        for (int tries = 0 ; tries < 1000 ; ++tries) {  // 循环1000次
            int x = random.nextInt(s.size());  // 生成随机的x坐标
            int y = random.nextInt(s.size());  // 生成随机的y坐标
            int orient = random.nextInt(4);  // 生成随机的方向

            if (place(s, x, y, orient)) return;  // 如果成功放置船，则返回
        }

        throw new RuntimeException("Could not place any more ships");  // 如果无法再放置船，则抛出异常
    // Attempt to fit the ship into the sea, starting from a given position and
    // in a given direction
    // This is by far the most complicated part of the program.
    // It will start at the position provided, and attempt to occupy tiles in the
    // requested direction. If it does not fit, either because of the edge of the
    // sea, or because of ships already in place, it will try to extend the ship
    // in the opposite direction instead. If that is not possible, it fails.
    public boolean place(Sea s, int x, int y, int orient) {
        if (placed) {
            throw new RuntimeException("Program error - placed ship " + id + " twice");
        }
        switch(orient) {
        case ORIENT_E:                 // east is increasing X coordinate
            orientX = 1; orientY = 0;  // Set the orientation for east direction
            break;
        case ORIENT_SE:                // southeast is increasing X and Y
            orientX = 1; orientY = 1;  // Set the orientation for southeast direction
            break;
        case ORIENT_S:                 // 南方增加Y坐标
            orientX = 0; orientY = 1;  // 设置X和Y方向的增量
            break;
        case ORIENT_SW:                // 西南方增加Y坐标但减少X坐标
            orientX = -1; orientY = 1; // 设置X和Y方向的增量
            break;
        default:
            throw new RuntimeException("Invalid orientation " + orient);  // 抛出异常，表示方向无效
        }

        if (!s.isEmpty(x, y)) return false; // 起始位置已被占据 - 放置失败

        startX = x; startY = y;  // 设置起始位置
        int tilesPlaced = 1;  // 已放置的瓷砖数量
        int nextX = startX;  // 下一个X坐标
        int nextY = startY;  // 下一个Y坐标
        while (tilesPlaced < size) {  // 当放置的瓷砖数量小于总数量时
            if (extendShip(s, nextX, nextY, nextX + orientX, nextY + orientY)) {  // 如果可以向前延伸船只
                // It is clear to extend the ship forwards
                tilesPlaced += 1;  // 放置的瓷砖数量加1
                nextX = nextX + orientX;  // 更新下一个 X 坐标
                nextY = nextY + orientY;  // 更新下一个 Y 坐标
            } else {
                int backX = startX - orientX;  // 计算后退时的 X 坐标
                int backY = startY - orientY;  // 计算后退时的 Y 坐标

                if (extendShip(s, startX, startY, backX, backY)) {  // 如果可以将船向后移动，使其变长
                    // We can move the ship backwards, so it can be one tile longer
                    tilesPlaced +=1;  // 放置瓦片数量加一
                    startX = backX;  // 更新起始 X 坐标
                    startY = backY;  // 更新起始 Y 坐标
                } else {
                    // Could not make it longer or move it backwards
                    return false;  // 无法使船变长或向后移动，返回 false
                }
            }
        }

        // Mark in the sea which tiles this ship occupies
        for (int i = 0; i < size; ++i) {  // 标记船占据的瓦片在海中的位置
            int sx = startX + i * orientX;  // 计算船的起始 x 坐标
            int sy = startY + i * orientY;  // 计算船的起始 y 坐标
            s.set(sx, sy, id);  // 在海域中设置船的位置
        }

        placed = true;  // 标记船已经放置
        return true;  // 返回放置成功
    }

    // 检查已经占据“from”坐标的船是否也可以占据“to”坐标
    // 它们必须在海域内，为空，并且不会导致船越过另一艘船
    private boolean extendShip(Sea s, int fromX, int fromY, int toX, int toY) {
        if (!s.isEmpty(toX, toY)) return false;  // 没有空间
        if ((fromX == toX)||(fromY == toY)) return true;  // 水平或垂直

        // 我们可以扩展船而不发生碰撞，但我们是在对角线上
        // 两艘船不应该可能在对角线上相交
// 检查在这里交叉的两个方块 - 如果其中一个为空，我们就没问题
// 如果它们都包含不同的船只，我们就没问题
// 但如果它们都包含相同的船只，我们就会交叉！
int corner1 = s.get(fromX, toY); // 获取坐标(fromX, toY)处的值
int corner2 = s.get(toX, fromY); // 获取坐标(toX, fromY)处的值
if ((corner1 == 0) || (corner1 != corner2)) return true; // 如果corner1为0或者corner1不等于corner2，则返回true
return false; // 否则返回false
}
```