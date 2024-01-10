# `basic-computer-games\09_Battle\java\Input.java`

```
# 导入需要的类
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.text.NumberFormat;

# 这个类处理从玩家读取的输入
# 每个输入都是一个 x 和 y 坐标
# 例如 5,3
public class Input {
    # 用于读取输入的缓冲读取器
    private BufferedReader reader;
    # 用于解析输入的数字格式化对象
    private NumberFormat parser;
    # 海域的大小，用于验证输入
    private int scale;
    # 输入是否结束的标志
    private boolean isQuit;
    # 最后读取的坐标
    private int[] coords;

    # 构造函数，初始化海域大小，并创建输入流和数字格式化对象
    public Input(int seaSize) {
        scale = seaSize;
        reader = new BufferedReader(new InputStreamReader(System.in));
        parser = NumberFormat.getIntegerInstance();
    }
    // 读取坐标信息的方法，抛出 IOException 异常
    public boolean readCoordinates() throws IOException {
        // 无限循环，等待用户输入
        while (true) {
            // 打印提示信息，要求用户输入目标 x,y 坐标
            System.out.print("\nTarget x,y\n> ");
            // 读取用户输入的一行数据
            String inputLine = reader.readLine();
            // 如果输入行为空，表示输入流结束，游戏无法继续，设置退出标志并返回 false
            if (inputLine == null) {
                System.out.println("\nGame quit\n");
                isQuit = true;
                return false;
            }

            // 将输入行按逗号分割成两个字段
            String[] fields = inputLine.split(",");
            // 如果分割后的字段数量不是两个，提示用户需要输入两个由逗号分隔的坐标
            if (fields.length != 2) {
                System.out.println("Need two coordinates separated by ','");
                continue;
            }

            // 初始化坐标数组
            coords = new int[2];
            boolean error = false;
            // 遍历两个字段，检查是否为合法的坐标值
            try {
                for (int c = 0 ; c < 2; ++c ) {
                    // 将字段转换为整数，并去除首尾空格
                    int val = Integer.parseInt(fields[c].strip());
                    // 如果值不在 1 到海域大小之间，提示用户坐标必须在 1 到海域大小之间
                    if ((val < 1) || (val > scale)) {
                        System.out.println("Coordinates must be from 1 to " + scale);
                        error = true;
                    } else {
                        coords[c] = val;
                    }
                }
            }
            // 捕获转换整数时的异常，提示用户坐标必须为数字
            catch (NumberFormatException ne) {
                System.out.println("Coordinates must be numbers");
                error = true;
            }
            // 如果没有错误，返回 true
            if (!error) return true;
        }
    }

    // 获取 x 坐标的方法
    public int x() { return coords[0]; }
    // 获取 y 坐标的方法
    public int y() { return coords[1]; }
# 闭合前面的函数定义
```