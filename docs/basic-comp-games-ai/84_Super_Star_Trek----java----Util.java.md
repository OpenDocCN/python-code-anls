# `basic-computer-games\84_Super_Star_Trek\java\Util.java`

```
// 导入所需的 Java 类
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * 为超级星际迷航游戏提供便利的工具方法
 */
public class Util {

    // 创建一个静态的随机数生成器对象
    static final Random random = new Random();

    // 生成一个随机浮点数
    public static float random() {
        return random.nextFloat();
    }

    // 生成一个从1到8的随机整数（包括1和8）
    public static int fnr() {    // 475
        // 生成一个从1到8的随机整数
        return toInt(random() * 7 + 1);
    }

    // 将浮点数转换为整数
    public static int toInt(final double num) {
        int x = (int) Math.floor(num);
        // 如果结果小于0，则取绝对值
        if (x < 0) x *= -1;
        return x;
    }

    // 打印输出字符串并换行
    public static void println(final String s) {
        System.out.println(s);
    }

    // 打印输出字符串
    public static void print(final String s) {
        System.out.print(s);
    }

    // 生成指定数量的空格字符串
    public static String tab(final int n) {
        return IntStream.range(1, n).mapToObj(num -> " ").collect(Collectors.joining());
    }

    // 获取字符串的长度
    public static int strlen(final String s) {
        return s.length();
    }

    // 从控制台输入字符串
    public static String inputStr(final String message) {
        System.out.print(message + "? ");
        // 创建一个从控制台读取输入的 BufferedReader 对象
        final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        try {
            // 读取输入的字符串并返回
            return reader.readLine();
        } catch (IOException ioe) {
            // 捕获输入输出异常并打印错误信息
            ioe.printStackTrace();
            return "";
        }
    }
}
    // 从用户输入中获取坐标，返回一个包含 x 和 y 坐标的整数数组
    public static int[] inputCoords(final String message) {
        // 循环直到得到有效的输入
        while (true) {
            // 获取用户输入的字符串
            final String input = inputStr(message);
            try {
                // 将输入字符串按逗号分割
                final String[] splitInput = input.split(",");
                // 如果分割后的数组长度为2
                if (splitInput.length == 2) {
                    // 将分割后的字符串转换为整数，并返回包含 x 和 y 坐标的数组
                    int x = Integer.parseInt(splitInput[0]);
                    int y = Integer.parseInt(splitInput[1]);
                    return new int[]{x, y};
                }
            } catch (Exception e) {
                // 捕获异常并打印堆栈信息
                e.printStackTrace();
            }
        }
    }

    // 从用户输入中获取浮点数
    public static float inputFloat(final String message) {
        // 循环直到得到有效的输入
        while (true) {
            // 打印提示信息
            System.out.print(message + "? ");
            // 从标准输入流中读取数据
            final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            try {
                // 读取用户输入的字符串
                final String input = reader.readLine();
                // 如果输入的字符串长度大于0，则将其转换为浮点数并返回
                if (input.length() > 0) {
                    return Float.parseFloat(input);
                }
            } catch (Exception e) {
                // 捕获异常并打印堆栈信息
                e.printStackTrace();
            }
        }
    }

    // 返回字符串的左侧指定长度的子字符串
    public static String leftStr(final String input, final int len) {
        // 如果输入为null或长度小于指定长度，则返回原字符串
        if (input == null || input.length() < len) return input;
        // 返回左侧指定长度的子字符串
        return input.substring(0, len);
    }

    // 返回字符串的中间指定长度的子字符串
    public static String midStr(final String input, final int start, final int len) {
        // 如果输入为null或长度小于起始位置加指定长度，则返回原字符串
        if (input == null || input.length() < ((start - 1) + len)) return input;
        // 返回中间指定长度的子字符串
        return input.substring(start - 1, (start - 1) + len);
    }

    // 返回字符串的右侧指定长度的子字符串
    public static String rightStr(final String input, final int len) {
        // 如果输入为null或长度小于指定长度，则返回空字符串
        if (input == null || input.length() < len) return "";
        // 返回右侧指定长度的子字符串
        return input.substring(input.length() - len);
    }

    // 对浮点数进行四舍五入
    public static double round(double value, int places) {
        // 如果小数位数小于0，则抛出异常
        if (places < 0) throw new IllegalArgumentException();
        // 创建 BigDecimal 对象并进行四舍五入
        BigDecimal bd = new BigDecimal(Double.toString(value));
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
# 闭合前面的函数定义
```