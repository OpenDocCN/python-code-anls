# `75_Roulette\java\oop\Wheel.java`

```
import java.util.Arrays; // 导入 Arrays 类，用于操作数组
import java.util.HashSet; // 导入 HashSet 类，用于创建集合
import java.util.Random; // 导入 Random 类，用于生成随机数

// 轮盘
public class Wheel {
    // 列出黑色的数字
    private HashSet<Integer> black = new HashSet<>(Arrays.asList(new Integer[] { 1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36 })); // 创建包含黑色数字的集合

    private Random random = new Random(); // 创建 Random 对象
    private int pocket = 38; // 初始化 pocket 变量为 38

    public static final int ZERO=0; // 常量 ZERO 被赋值为 0
    public static final int BLACK=1; // 常量 BLACK 被赋值为 1
    public static final int RED=2; // 常量 RED 被赋值为 2

    // 设置轮盘。您可以调用 "spin"，然后检查结果。
    public Wheel() { // 构造函数
    }
    // 作弊/测试模式
    void setSeed(long l) {
        random.setSeed(l); // 设置随机数种子
    }

    // 旋转轮盘到一个新的随机值
    public void spin() {
        // 持续旋转一段时间
        do {
            try {
                // 1秒延迟。在哪里停下来，谁也不知道
                Thread.sleep(1000); // 线程休眠1秒
            }
            catch (InterruptedException e) {}

            pocket = random.nextInt(38) + 1; // 生成一个1到38之间的随机数
        } while (random.nextInt(4) > 0); // 保持旋转直到停止
    }

    // 数字的字符串表示；1-36，0，或00
    public String value() {
        // 如果口袋里的数字是37，则返回字符串"0"
        if (pocket == 37) return "0";
        // 如果口袋里的数字是38，则返回字符串"00"
        else if (pocket == 38) return "00";
        // 否则返回口袋里的数字的字符串形式
        else return String.valueOf(pocket);
    }

    // 如果出现了0或00，则返回true
    public boolean zero() {
        // 如果口袋里的数字大于36，则返回true
        return (pocket > 36);
    }

    // 如果出现了除0和00之外的数字，则返回true
    public boolean isNumber() {
        // 如果口袋里的数字小于37，则返回true
        return (pocket < 37);
    }

    // 掷出的数字
    public int number() {
        // 如果出现了0，则返回0
        if (zero()) return 0;
        // 否则返回口袋里的数字
        else return pocket;
    }
    }

    // 定义一个公共方法，返回整数类型的颜色值，可能是 ZERO、BLACK 或 RED
    public int color() {
        // 如果是 ZERO，返回 ZERO
        if (zero()) return ZERO;
        // 如果是黑色，返回 BLACK
        else if (black.contains(pocket)) return BLACK;
        // 否则返回 RED
        else return RED;
    }
}
```