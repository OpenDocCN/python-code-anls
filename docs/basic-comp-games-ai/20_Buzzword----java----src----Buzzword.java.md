# `basic-computer-games\20_Buzzword\java\src\Buzzword.java`

```

# 导入 Scanner 类
import java.util.Scanner;

public class Buzzword {

    public static void main(final String[] args) {
        # 使用 try-with-resources 语句，创建 Scanner 对象
        # Scanner 是 Closeable 接口的实现类，所以在程序结束前必须关闭
        try (
            final Scanner scanner = new Scanner(System.in);
        ) {
            # 创建 BuzzwordSupplier 对象
            final BuzzwordSupplier buzzwords = new BuzzwordSupplier();
            # 创建 UserInterface 对象，传入 Scanner、System.out 和 buzzwords
            final UserInterface userInterface = new UserInterface(
                    scanner, System.out, buzzwords);
            # 运行用户界面
            userInterface.run();
        }
    }
}

```