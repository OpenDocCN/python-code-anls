# `basic-computer-games\93_23_Matches\javascript\23matches.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的值传递给 resolve 函数
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主控制部分，使用 async 函数定义
async function main()
{
    // 打印游戏标题和介绍
    print(tab(31) + "23 MATCHES\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print(" THIS IS A GAME CALLED '23 MATCHES'.\n");
    print("\n");
    print("WHEN IT IS YOUR TURN, YOU MAY TAKE ONE, TWO, OR THREE\n");
    print("MATCHES. THE OBJECT OF THE GAME IS NOT TO HAVE TO TAKE\n");
    print("THE LAST MATCH.\n");
    print("\n");
    print("LET'S FLIP A COIN TO SEE WHO GOES FIRST.\n");
    print("IF IT COMES UP HEADS, I WILL WIN THE TOSS.\n");

    // 初始化游戏参数
    n = 23;
    q = Math.floor(2 * Math.random());
    if (q != 1) {
        print("TAILS! YOU GO FIRST. \n");
        print("\n");
    } else {
        print("HEADS! I WIN! HA! HA!\n");
        print("PREPARE TO LOSE, MEATBALL-NOSE!!\n");
        print("\n");
        print("I TAKE 2 MATCHES\n");
        n -= 2;
    }

    // 游戏循环
    while (1) {
        if (q == 1) {
            print("THE NUMBER OF MATCHES IS NOW " + n + "\n");
            print("\n");
            print("YOUR TURN -- YOU MAY TAKE 1, 2 OR 3 MATCHES.\n");
        }
        print("HOW MANY DO YOU WISH TO REMOVE ");
        while (1) {
            k = parseInt(await input());
            if (k <= 0 || k > 3) {
                print("VERY FUNNY! DUMMY!\n");
                print("DO YOU WANT TO PLAY OR GOOF AROUND?\n");
                print("NOW, HOW MANY MATCHES DO YOU WANT ");
            } else {
                break;
            }
        }
        n -= k;
        print("THERE ARE NOW " + n + " MATCHES REMAINING.\n");
        if (n == 4) {
            z = 3;
        } else if (n == 3) {
            z = 2;
        } else if (n == 2) {
            z = 1;
        } else if (n > 1) {
            z = 4 - k;
        } else {
            print("YOU WON, FLOPPY EARS !\n");
            print("THINK YOU'RE PRETTY SMART !\n");
            print("LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!\n");
            break;
        }
        print("MY TURN ! I REMOVE " + z + " MATCHES\n");
        n -= z;
        if (n <= 1) {
            print("\n");
            print("YOU POOR BOOB! YOU TOOK THE LAST MATCH! I GOTCHA!!\n");
            print("HA ! HA ! I BEAT YOU !!!\n");
            print("\n");
            print("GOOD BYE LOSER!\n");
            break;
        }
        q = 1;
    }

}

// 调用主函数开始游戏
main();

```