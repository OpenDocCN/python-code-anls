# `d:/src/tocomm/basic-computer-games\05_Bagels\java\Bagels.java`

```
# 打开一个 ZIP 文件并读取其中的内容，返回文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 以二进制模式打开文件，并将其内容封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 用字节流内容创建一个 ZIP 对象，'r'表示以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 通过遍历 ZIP 对象的文件名列表，读取每个文件的数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
    // Intro text
    System.out.println("\n\n                Bagels"); // 打印游戏标题
    System.out.println("Creative Computing  Morristown, New Jersey"); // 打印游戏信息
    System.out.println("\n\n");
    System.out.print("Would you like the rules (Yes or No)? "); // 打印询问是否需要游戏规则

    // Need instructions?
    Scanner scan = new Scanner(System.in); // 创建一个用于接收用户输入的 Scanner 对象
    String s = scan.nextLine(); // 读取用户输入的字符串
    if (s.length() == 0 || s.toUpperCase().charAt(0) != 'N') { // 判断用户是否需要游戏规则
      System.out.println();
      System.out.println("I am thinking of a three-digit number.  Try to guess"); // 打印游戏提示
      System.out.println("my number and I will give you clues as follows:"); // 打印游戏提示
      System.out.println("   PICO   - One digit correct but in the wrong position"); // 打印游戏提示
      System.out.println("   FERMI  - One digit correct and in the right position"); // 打印游戏提示
      System.out.println("   BAGELS - No digits correct"); // 打印游戏提示
    }

    // Loop for playing multiple games
    boolean stillPlaying = true; // 创建一个布尔变量，用于控制游戏循环
    while(stillPlaying) { // 当游戏仍在进行时执行以下操作

      // Set up a new game
      BagelGame game = new BagelGame(); // 创建一个新的BagelGame对象
      System.out.println("\nO.K.  I have a number in mind."); // 打印消息

      // Loop guess and responsses until game is over
      while (!game.isOver()) { // 当游戏未结束时循环猜测和回应
        String guess = getValidGuess(game); // 获取有效的猜测
        String response = game.makeGuess(guess); // 进行猜测并获取回应
        // Don't print a response if the game is won
        if (!game.isWon()) { // 如果游戏未赢则打印回应
          System.out.println(response);
        }
      }

      // Game is over. But did we win or lose?
      if (game.isWon()) { // 如果游戏赢了
        System.out.println("You got it!!!\n"); // 打印消息
        gamesWon++; // 游戏赢的次数加一
      } else {
        // 如果猜测次数用完，打印消息
        System.out.println("Oh well");
        // 打印消息，显示猜测次数
        System.out.print("That's " + BagelGame.MAX_GUESSES + " guesses.  ");
        // 打印消息，显示正确数字
        System.out.println("My number was " + game.getSecretAsString());
      }

      // 获取玩家是否要重新开始游戏的响应
      stillPlaying = getReplayResponse();
    }

    // 打印结束游戏的消息
    if (gamesWon > 0) {
      System.out.println("\nA " + gamesWon + " point Bagels buff!!");
    }
    // 打印结束游戏的消息
    System.out.println("Hope you had fun.  Bye.\n");
  }

  // 获取有效的猜测
  private static String getValidGuess(BagelGame game) {
    // 持续询问直到得到有效的猜测
    Scanner scan = new Scanner(System.in);
    boolean valid = false;
    # 初始化猜测字符串和错误信息字符串
    String guess = "";
    String error;

    # 当猜测不合法时循环
    while (!valid) {
      # 打印猜测次数并获取用户输入的猜测字符串
      System.out.print("Guess # " + game.getGuessNum() + "     ? ");
      guess = scan.nextLine().trim();
      # 验证猜测字符串是否合法，并将错误信息赋值给error变量
      error = game.validateGuess(guess);
      # 如果错误信息为空，则猜测合法，跳出循环
      if (error == "") {
        valid = true;
      } else {
        # 如果有错误信息，则打印错误信息
        System.out.println(error);
      }
    }
    # 返回猜测字符串
    return guess;
  }

  # 获取用户是否想要重新开始游戏的响应
  private static boolean getReplayResponse() {
    # 创建一个用于从控制台获取输入的Scanner对象
    Scanner scan = new Scanner(System.in);
    # 循环直到输入了非空字符串
    # 用户输入的字符串长度为0时，继续循环
    while (true):  # 进入一个无限循环
      System.out.print("Play again (Yes or No)? ");  # 打印提示信息，询问用户是否要再玩一次
      String response = scan.nextLine().trim();  # 从用户输入中获取一行字符串，并去除首尾空格
      if (response.length() > 0):  # 如果用户输入的字符串长度大于0
        return response.toUpperCase().charAt(0) == 'Y';  # 将用户输入的字符串转换为大写后，取第一个字符，判断是否为'Y'，然后返回对应的布尔值
      # 如果用户输入的字符串长度为0，则继续循环
    }
  }
}
```