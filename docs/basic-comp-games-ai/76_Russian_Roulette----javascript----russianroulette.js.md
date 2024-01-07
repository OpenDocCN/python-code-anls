# `basic-computer-games\76_Russian_Roulette\javascript\russianroulette.js`

```

// 定义一个打印函数，用于在页面上输出文本
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
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 在页面上输出提示符
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

// 主程序，使用 async 函数定义，可以使用 await 关键字等待 Promise 对象的解析
async function main()
{
    // 输出游戏标题
    print(tab(28) + "RUSSIAN ROULETTE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS A GAME OF >>>>>>>>>>RUSSIAN ROULETTE.\n");
    restart = true;
    while (1) {
        if (restart) {
            restart = false;
            print("\n");
            print("HERE IS A REVOLVER.\n");
        }
        print("TYPE '1' TO SPIN CHAMBER AND PULL TRIGGER.\n");
        print("TYPE '2' TO GIVE UP.\n");
        print("GO");
        n = 0;
        while (1) {
            // 等待输入，使用 parseInt 将输入的字符串转换为整数
            i = parseInt(await input());
            if (i == 2) {
                print("     CHICKEN!!!!!\n");
                break;
            }
            n++;
            // 使用 Math.random() 生成随机数，如果大于 0.833333，则表示触发了“死亡”，游戏结束
            if (Math.random() > 0.833333) {
                print("     BANG!!!!!   YOU'RE DEAD!\n");
                print("CONDOLENCES WILL BE SENT TO YOUR RELATIVES.\n");
                break;
            }
            // 如果尝试次数超过 10 次，则表示玩家获胜
            if (n > 10) {
                print("YOU WIN!!!!!\n");
                print("LET SOMEONE ELSE BLOW HIS BRAINS OUT.\n");
                restart = true;
                break;
            }
            print("- CLICK -\n");
            print("\n");
        }
        print("\n");
        print("\n");
        print("\n");
        print("...NEXT VICTIM...\n");
    }
}

// 调用主程序
main();

```