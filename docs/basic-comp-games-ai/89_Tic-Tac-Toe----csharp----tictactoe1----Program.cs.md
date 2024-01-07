# `basic-computer-games\89_Tic-Tac-Toe\csharp\tictactoe1\Program.cs`

```

// 打印文本在屏幕上，文本前面有30个空格
Console.WriteLine("TIC TAC TOE".PadLeft(30));
// 打印文本在屏幕上，文本前面有15个空格
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
    while(true) {
        Console.Write("YOUR MOVE?");
        string input = Console.ReadLine();
        if (int.TryParse(input, out int number)) {
            return number;
        }
    }
}

// 走法函数
int move(int number) {
    return number - 8 * (int)((number - 1) / 8);
}

```