# `basic-computer-games\89_Tic-Tac-Toe\csharp\tictactoe1\Program.cs`

```
// 在屏幕上打印文本，文本前面有30个空格
Console.WriteLine("TIC TAC TOE".PadLeft(30));
// 在屏幕上打印文本，文本前面有15个空格
Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(15));
// 在屏幕上打印三个空行
Console.WriteLine("\n\n\n");
// 这个程序玩井字棋
// 机器先走
Console.WriteLine("THE GAME BOARD IS NUMBERED:\n");
Console.WriteLine("1  2  3");
Console.WriteLine("8  9  4");
Console.WriteLine("7  6  5");

// 主程序
while(true) {
    int a, b, c, d, e;
    int p, q, r, s;
    a = 9;
    Console.WriteLine("\n\n");
    computerMoves(a);
    p = readYourMove();
    b = move(p + 1);
    computerMoves(b);
    q = readYourMove();
    if (q == move(b + 4)) {
        c = move(b + 2);
        computerMoves(c);
        r = readYourMove();
        if (r == move(c + 4)) {
            if (p % 2 != 0) {
                d = move(c + 3);
                computerMoves(d);
                s = readYourMove();
                if (s == move(d + 4)) {
                    e = move(d + 6);
                    computerMoves(e);
                    Console.WriteLine("THE GAME IS A DRAW.");
                } else {
                    e = move(d + 4);
                    computerMoves(e);
                    Console.WriteLine("AND WINS ********");
                }
            } else {
                d = move(c + 7);
                computerMoves(d);
                Console.WriteLine("AND WINS ********");
            }
        } else {
            d = move(c + 4);
            computerMoves(d);
            Console.WriteLine("AND WINS ********");
        }
    } else {
        c = move(b + 4);
        computerMoves(c);
        Console.WriteLine("AND WINS ********");
    }
}

// 机器走的函数
void computerMoves(int move) {
        Console.WriteLine("COMPUTER MOVES " + move);
}
// 读取你的走法
int readYourMove() {
    # 进入循环，等待用户输入
    while(true) {
        # 提示用户输入他们的移动
        Console.Write("YOUR MOVE?");
        # 读取用户输入的字符串
        string input = Console.ReadLine();
        # 尝试将输入的字符串转换为整数，如果成功则返回该整数
        if (int.TryParse(input, out int number)) {
            return number;
        }
    }
# 定义一个名为move的函数，接受一个整数参数
int move(int number) {
    # 返回参数减去8的倍数，即取余数为1时的值
    return number - 8 * (int)((number - 1) / 8);
}
```