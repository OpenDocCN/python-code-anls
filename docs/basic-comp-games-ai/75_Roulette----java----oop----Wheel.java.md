# `basic-computer-games\75_Roulette\java\oop\Wheel.java`

```
// 导入必要的类
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

// 轮盘赌轮
public class Wheel {
    // 列出黑色的数字
    private HashSet<Integer> black = new HashSet<>(Arrays.asList(new Integer[] { 1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36 }));

    private Random random = new Random();
    private int pocket = 38;

    public static final int ZERO=0;
    public static final int BLACK=1;
    public static final int RED=2;

    // 设置轮盘。您可以调用 "spin"，然后检查结果。
    public Wheel() {
    }

    // 作弊/测试模式
    void setSeed(long l) {
        random.setSeed(l);
    }

    // 将轮盘旋转到新的随机值。
    public void spin() {
        // 不断旋转一段时间
        do {
            try {
                // 1秒延迟。停在哪里，谁也不知道
                Thread.sleep(1000);
            }
            catch (InterruptedException e) {}

            pocket = random.nextInt(38) + 1;
        } while (random.nextInt(4) > 0); // 保持旋转直到停止
    }

    // 数字的字符串表示；1-36，0或00
    public String value() {
        if (pocket == 37) return "0";
        else if (pocket == 38) return "00";
        else return String.valueOf(pocket);
    }

    // 如果命中了0或00，则为真
    public boolean zero() {
        return (pocket > 36);
    }

    // 如果命中的是0或00之外的任何数字，则为真
    public boolean isNumber() {
        return (pocket < 37);
    }

    // 掷出的数字
    public int number() {
        if (zero()) return 0;
        else return pocket;
    }

    // ZERO、BLACK或RED
    public int color() {
        if (zero()) return ZERO;
        else if (black.contains(pocket)) return BLACK;
        else return RED;
    }
}
```