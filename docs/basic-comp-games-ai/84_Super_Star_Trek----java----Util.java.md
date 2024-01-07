# `basic-computer-games\84_Super_Star_Trek\java\Util.java`

```

// 导入所需的 Java 类
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
importjava.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * 为 Super Star Trek 游戏提供便利的工具方法
 */
public class Util {

    // 创建一个随机数生成器对象
    static final Random random = new Random();

    // 生成一个随机浮点数
    public static float random() {
        return random.nextFloat();
    }

    // 生成一个从 1 到 8 的随机整数
    public static int fnr() {    
        return toInt(random() * 7 + 1);
    }

    // 将 double 类型转换为整数
    public static int toInt(final double num) {
        int x = (int) Math.floor(num);
        if (x < 0) x *= -1;
        return x;
    }

    // 打印字符串并换行
    public static void println(final String s) {
        System.out.println(s);
    }

    // 打印字符串
    public static void print(final String s) {
        System.out.print(s);
    }

    // 生成指定数量的空格字符串
    public static String tab(final int n) {
        return IntStream.range(1, n).mapToObj(num -> " ").collect(Collectors.joining());
    }

    // 返回字符串的长度
    public static int strlen(final String s) {
        return s.length();
    }

    // 从控制台输入字符串
    public static String inputStr(final String message) {
        // 提示用户输入信息
        System.out.print(message + "? ");
        // 创建输入流对象
        final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        try {
            // 读取用户输入的字符串
            return reader.readLine();
        } catch (IOException ioe) {
            ioe.printStackTrace();
            return "";
        }
    }

    // 从控制台输入坐标数组
    public static int[] inputCoords(final String message) {
        while (true) {
            final String input = inputStr(message);
            try {
                final String[] splitInput = input.split(",");
                if (splitInput.length == 2) {
                    int x = Integer.parseInt(splitInput[0]);
                    int y = Integer.parseInt(splitInput[0]);
                    return new int[]{x, y};
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    // 从控制台输入浮点数
    public static float inputFloat(final String message) {
        while (true) {
            System.out.print(message + "? ");
            final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            try {
                final String input = reader.readLine();
                if (input.length() > 0) {
                    return Float.parseFloat(input);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    // 返回字符串的左侧子串
    public static String leftStr(final String input, final int len) {
        if (input == null || input.length() < len) return input;
        return input.substring(0, len);
    }

    // 返回字符串的中间子串
    public static String midStr(final String input, final int start, final int len) {
        if (input == null || input.length() < ((start - 1) + len)) return input;
        return input.substring(start - 1, (start - 1) + len);
    }

    // 返回字符串的右侧子串
    public static String rightStr(final String input, final int len) {
        if (input == null || input.length() < len) return "";
        return input.substring(input.length() - len);
    }

    // 对 double 类型的数值进行四舍五入
    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();
        BigDecimal bd = new BigDecimal(Double.toString(value));
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}

```