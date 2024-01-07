# `basic-computer-games\47_Hi-Lo\javascript\hi-lo.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象，当用户输入完成时，Promise 对象状态变为 resolved
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听用户输入，当用户按下回车键时，将输入的值传递给 resolve 函数
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

// 主程序，使用 async/await 实现异步操作
async function main()
{
    // 输出游戏标题和介绍
    print(tab(34) + "HI LO\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS THE GAME OF HI LO.\n");
    print("\n");
    print("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE\n");
    print("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU\n");
    print("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!\n");
    print("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,\n");
    print("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.\n");
    print("\n");
    
    // 初始化变量
    r = 0;
    
    // 游戏循环
    while (1) {
        b = 0;
        print("\n");
        // 生成一个 0 到 100 之间的随机数
        y = Math.floor(100 * Math.random());
        for (b = 1; b <= 6; b++) {
            print("YOUR GUESS");
            // 等待用户输入，并将输入的字符串转换为整数
            a = parseInt(await input());
            if (a < y) {
                print("YOUR GUESS IS TOO LOW.\n");
            } else if (a > y) {
                print("YOUR GUESS IS TOO HIGH.\n");
            } else {
                break;
            }
            print("\n");
        }
        if (b > 6) {
            print("YOU BLEW IT...TOO BAD...THE NUMBER WAS " + y + "\n");
            r = 0;
        } else {
            print("GOT IT!!!!!!!!!!   YOU WIN " + y + " DOLLARS.\n");
            r += y;
            print("YOUR TOTAL WINNINGS ARE NOW " + r + " DOLLARS.\n");
        }
        print("\n");
        print("PLAY AGAIN (YES OR NO)");
        // 等待用户输入，并将输入的字符串转换为大写
        str = await input();
        str = str.toUpperCase();
        if (str != "YES")
            break;
    }
    print("\n");
    print("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!\n");
}

// 调用主程序
main();

```