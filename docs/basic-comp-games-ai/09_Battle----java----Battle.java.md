# `basic-computer-games\09_Battle\java\Battle.java`

```py
import java.io.IOException;  // 导入处理输入输出异常的类
import java.util.ArrayList;  // 导入用于存储对象的动态数组类
import java.util.Arrays;      // 导入用于操作数组的类
import java.util.Collections; // 导入用于操作集合的类
import java.util.Comparator;  // 导入用于对象比较的接口
import java.util.Random;      // 导入生成随机数的类
import java.util.function.Predicate;  // 导入用于定义条件的函数式接口
import java.text.NumberFormat;       // 导入用于格式化数字的类

/* This class holds the game state and the game logic */
public class Battle {

    /* parameters of the game */
    private int seaSize;    // 游戏海域的大小
    private int[] sizes;    // 不同大小的船只
    private int[] counts;   // 每种大小船只的数量

    /* The game setup - the ships and the sea */
    private ArrayList<Ship> ships;  // 存储船只对象的动态数组
    private Sea sea;                // 海域对象

    /* game state counts */
    private int[] losses;  // 每种类型的船只被击沉的数量
    private int hits;      // 玩家击中的次数
    private int misses;    // 玩家未击中的次数

    // Names of ships of each size. The game as written has ships of size 3, 4 and 5 but
    // can easily be modified. It makes no sense to have a ship of size zero though.
    private static String NAMES_BY_SIZE[] = {  // 不同大小船只的名称
        "error",            // 错误情况
        "size1",            // 大小为1的船只
        "destroyer",        // 驱逐舰
        "cruiser",          // 巡洋舰
        "aircraft carrier", // 航空母舰
        "size5" };          // 大小为5的船只

    // Entrypoint
    public static void main(String args[]) {
        Battle game = new Battle(6,                        // 海域大小为6 x 6
                                 new int[] { 2, 3, 4 },    // 船只大小为2、3和4
                                 new int[] { 2, 2, 2 });   // 每种大小船只的数量为2
        game.play();  // 开始游戏
    }
}
    // 构造函数，初始化战斗游戏
    public Battle(int scale, int[] shipSizes, int[] shipCounts) {
        // 设置海域大小
        seaSize = scale;
        // 设置船只尺寸和数量
        sizes = shipSizes;
        counts = shipCounts;

        // 验证参数
        if (seaSize < 4) throw new RuntimeException("Sea Size " + seaSize + " invalid, must be at least 4");

        // 验证船只尺寸是否合法
        for (int sz : sizes) {
            if ((sz < 1) || (sz > seaSize))
                throw new RuntimeException("Ship has invalid size " + sz);
        }

        // 验证船只数量和尺寸数量是否匹配
        if (counts.length != sizes.length) {
            throw new RuntimeException("Ship counts must match");
        }

        // 初始化游戏状态
        sea = new Sea(seaSize);          // 存储每个方格上的船只信息
        ships = new ArrayList<Ship>();   // 存储所有船只的位置和状态
        losses = new int[counts.length]; // 每种类型船只被击沉的数量

        // 构建所有船只的列表
        int shipNumber = 1;
        for (int type = 0; type < counts.length; ++type) {
            for (int i = 0; i < counts[i]; ++i) {
                ships.add(new Ship(shipNumber++, sizes[type]));
            }
        }

        // 将船只按照大小降序排列，以便先放置大船只
        ArrayList<Ship> largestFirst = new ArrayList<>(ships);
        Collections.sort(largestFirst, Comparator.comparingInt((Ship ship) -> ship.size()).reversed());

        // 将每艘船只放入海域中
        for (Ship ship : largestFirst) {
            ship.placeRandom(sea);
        }
    }
# 闭合前面的函数定义
```