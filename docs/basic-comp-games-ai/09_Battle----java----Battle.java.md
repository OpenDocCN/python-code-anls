# `09_Battle\java\Battle.java`

```
import java.io.IOException;  // 导入处理输入输出异常的类
import java.util.ArrayList;  // 导入使用动态数组的类
import java.util.Arrays;  // 导入处理数组的类
import java.util.Collections;  // 导入处理集合的类
import java.util.Comparator;  // 导入比较器的类
import java.util.Random;  // 导入生成随机数的类
import java.util.function.Predicate;  // 导入函数式接口，用于定义条件
import java.text.NumberFormat;  // 导入格式化数字的类

/* This class holds the game state and the game logic */
public class Battle {

    /* parameters of the game */
    private int seaSize;  // 游戏海域的大小
    private int[] sizes;  // 不同类型船只的大小
    private int[] counts;  // 不同类型船只的数量

    /* The game setup - the ships and the sea */
    private ArrayList<Ship> ships;  // 存储船只信息的动态数组
    private Sea sea; // 声明私有的Sea对象

    /* game state counts */
    private int[] losses;    // 每种类型的船只被击沉的数量
    private int hits;        // 玩家击中的次数
    private int misses;      // 玩家未击中的次数

    // 每种大小的船只的名称。游戏中有大小为3、4和5的船只，但可以很容易地修改。但是没有意义的是有大小为零的船只。
    private static String NAMES_BY_SIZE[] = {
        "error",
        "size1",
        "destroyer",
        "cruiser",
        "aircraft carrier",
        "size5" };

    // 入口点
    public static void main(String args[]) {
        Battle game = new Battle(6,                        // Sea is 6 x 6 tiles
                                 new int[] { 2, 3, 4 },    // 定义船只的大小为2、3、4
                                 new int[] { 2, 2, 2 });   // 每种大小的船只数量为2
        game.play();
    }

    public Battle(int scale, int[] shipSizes, int[] shipCounts) {
        seaSize = scale;  // 设置海域大小
        sizes = shipSizes;  // 设置船只大小数组
        counts = shipCounts;  // 设置船只数量数组

        // 验证参数
        if (seaSize < 4) throw new RuntimeException("Sea Size " + seaSize + " invalid, must be at least 4");  // 如果海域大小小于4，则抛出异常

        for (int sz : sizes) {
            if ((sz < 1) || (sz > seaSize))
                throw new RuntimeException("Ship has invalid size " + sz);  // 如果船只大小小于1或大于海域大小，则抛出异常
        }

        if (counts.length != sizes.length) {
            throw new RuntimeException("Ship counts must match");  // 如果船只数量数组长度与船只大小数组长度不匹配，则抛出异常
        // 初始化游戏状态
        sea = new Sea(seaSize);          // 保存每个方格上的船只信息
        ships = new ArrayList<Ship>();   // 保存所有船只的位置和状态
        losses = new int[counts.length]; // 每种类型的船只被击沉的数量

        // 构建所有船只的列表
        int shipNumber = 1;
        for (int type = 0; type < counts.length; ++type) {
            for (int i = 0; i < counts[i]; ++i) {
                ships.add(new Ship(shipNumber++, sizes[type]));
            }
        }

        // 当我们将船只放入海中时，我们先放入最大的船只，否则它们可能无法放下
        ArrayList<Ship> largestFirst = new ArrayList<>(ships);
        Collections.sort(largestFirst, Comparator.comparingInt((Ship ship) -> ship.size()).reversed());
        // 将每艘船放入海中
        for (Ship ship : largestFirst) {
            ship.placeRandom(sea);
        }
    }

    public void play() {
        System.out.println("The following code of the bad guys' fleet disposition\nhas been captured but not decoded:\n");
        System.out.println(sea.encodedDump());
        System.out.println("De-code it and use it if you can\nbut keep the de-coding method a secret.\n");

        int lost = 0;
        System.out.println("Start game");
        Input input = new Input(seaSize);
        try {
            while (lost < ships.size()) {          // 游戏在仍有未沉没的船只时继续进行
                if (! input.readCoordinates()) {   // ... 除非用户没有更多输入
                    return;
                }
                // 将用户输入的坐标转换为计算机内部的行列坐标
                int row = seaSize - input.y();
                int col = input.x() - 1;

                // 如果海面上指定位置为空，则增加misses计数，并输出提示信息
                if (sea.isEmpty(col, row)) {
                    ++misses;
                    System.out.println("Splash!  Try again.");
                } else {
                    // 获取海面上指定位置的船只对象
                    Ship ship = ships.get(sea.get(col, row) - 1);
                    // 如果船只已经被击沉，则增加misses计数，并输出提示信息
                    if (ship.isSunk()) {
                        ++misses;
                        System.out.println("There used to be a ship at that point, but you sunk it.");
                        System.out.println("Splash!  Try again.");
                    } else if (ship.wasHit(col, row)) {
                        // 如果船只在指定位置已经被击中，则增加misses计数，并输出提示信息
                        ++misses;
                        System.out.println("You already put a hole in ship number " + ship.id());
                        System.out.println("Splash!  Try again.");
                    } else {
                        // 如果船只在指定位置未被击中，则标记为击中
                        ship.hit(col, row);
                        ++hits;  // 命中次数加一
                        System.out.println("A direct hit on ship number " + ship.id());  // 打印直接命中船只编号的消息

                        // 如果击中了一艘船，我们需要知道它是否被击沉。
                        // 如果是，告诉玩家并更新我们的计数
                        if (ship.isSunk()) {  // 如果船只被击沉
                            ++lost;  // 被击沉的船只数量加一
                            System.out.println("And you sunk it.  Hurrah for the good guys.");  // 打印击沉船只的消息
                            System.out.print("So far, the bad guys have lost ");  // 打印迄今为止坏家伙们失去的船只数量
                            ArrayList<String> typeDescription = new ArrayList<>();  // 创建一个存储船只类型描述的列表
                            for (int i = 0 ; i < sizes.length; ++i) {  // 遍历船只尺寸数组
                                if (sizes[i] == ship.size()) {  // 如果船只尺寸与当前遍历的尺寸相同
                                    ++losses[i];  // 对应尺寸的失去数量加一
                                }
                                StringBuilder sb = new StringBuilder();  // 创建一个字符串构建器
                                sb.append(losses[i]);  // 添加失去的船只数量
                                sb.append(" ");  // 添加空格
                                sb.append(NAMES_BY_SIZE[sizes[i]]);  // 添加船只尺寸对应的名称
                                if (losses[i] != 1)  // 如果失去的船只数量不为1
                                    sb.append("s");  // 添加's'表示复数
                            typeDescription.add(sb.toString());  // 将字符串构建器中的内容添加到类型描述列表中
                            System.out.println(String.join(", ", typeDescription));  // 打印类型描述列表的内容
                            double ratioNum = ((double)misses)/hits;  // 计算未命中数与命中数的比率
                            String ratio = NumberFormat.getInstance().format(ratioNum);  // 格式化比率数值
                            System.out.println("Your current splash/hit ratio is " + ratio);  // 打印当前的命中/未命中比率

                            if (lost == ships.size()) {  // 如果失去的船只数量等于敌船总数
                                System.out.println("You have totally wiped out the bad guys' fleet");  // 打印已经消灭了敌人的全部舰队
                                System.out.println("With a final splash/hit ratio of " + ratio);  // 打印最终的命中/未命中比率

                                if (misses == 0) {  // 如果未命中数为0
                                    System.out.println("Congratulations - A direct hit every time.");  // 恭喜 - 每次都直接命中
                                }

                                System.out.println("\n****************************\n");  // 打印分隔线
                            }
                        }
                    }
                }
        }
        catch (IOException e) {
            // 如果发生 IOException 异常，打印错误信息
            // 这在控制台运行时不应该发生，但是 Java 要求我们检查它
            System.err.println("System error.\n" + e);
        }
    }
}
```