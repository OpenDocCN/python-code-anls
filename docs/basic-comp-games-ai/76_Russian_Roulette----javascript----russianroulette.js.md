# `basic-computer-games\76_Russian_Roulette\javascript\russianroulette.js`

```
// 定义一个打印函数，将字符串添加到输出元素中
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

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析输入的值
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序
async function main()
{
    // 打印游戏标题
    print(tab(28) + "RUSSIAN ROULETTE\n");
    // 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏说明
    print("THIS IS A GAME OF >>>>>>>>>>RUSSIAN ROULETTE.\n");
    // 设置重启标志为 true
    restart = true;
    # 进入无限循环，直到条件不满足
    while (1) {
        # 如果需要重新开始游戏，则重置标志位并打印提示信息
        if (restart) {
            restart = false;
            print("\n");
            print("HERE IS A REVOLVER.\n");
        }
        # 打印游戏选项
        print("TYPE '1' TO SPIN CHAMBER AND PULL TRIGGER.\n");
        print("TYPE '2' TO GIVE UP.\n");
        print("GO");
        # 初始化计数器
        n = 0;
        # 进入内层循环，直到条件不满足
        while (1) {
            # 获取用户输入并转换为整数
            i = parseInt(await input());
            # 如果用户选择放弃，则打印提示信息并跳出内层循环
            if (i == 2) {
                print("     CHICKEN!!!!!\n");
                break;
            }
            # 增加计数器
            n++;
            # 如果随机数大于0.833333，则打印游戏失败的提示信息并跳出内层循环
            if (Math.random() > 0.833333) {
                print("     BANG!!!!!   YOU'RE DEAD!\n");
                print("CONDOLENCES WILL BE SENT TO YOUR RELATIVES.\n");
                break;
            }
            # 如果计数器大于10，则打印游戏胜利的提示信息，并设置重新开始标志位，然后跳出内层循环
            if (n > 10) {
                print("YOU WIN!!!!!\n");
                print("LET SOMEONE ELSE BLOW HIS BRAINS OUT.\n");
                restart = true;
                break;
            }
            # 否则打印扳机声音的提示信息
            print("- CLICK -\n");
            print("\n");
        }
        # 打印游戏结束的提示信息
        print("\n");
        print("\n");
        print("\n");
        print("...NEXT VICTIM...\n");
    }
# 调用名为main的函数
main();
```