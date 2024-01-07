# `basic-computer-games\70_Poetry\javascript\poetry.js`

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
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的值添加到输出元素中，并解析 Promise
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

// 主程序，使用 async 函数定义
async function main()
{
    // 打印标题
    print(tab(30) + "POETRY\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");

    times = 0;

    i = 1;
    j = 1;
    k = 0;
    u = 0;
    // 无限循环
    while (1) {
        // 根据 j 的值选择不同的诗句
        if (j == 1) {
            switch (i) {
                case 1:
                    print("MIDNIGHT DREARY");
                    break;
                case 2:
                    print("FIERY EYES");
                    break;
                case 3:
                    print("BIRD OF FIEND");
                    break;
                case 4:
                    print("THING OF EVIL");
                    break;
                case 5:
                    print("PROPHET");
                    break;
            }
        } else if (j == 2) {
            switch (i) {
                case 1:
                    print("BEGUILING ME");
                    u = 2;
                    break;
                case 2:
                    print("THRILLED ME");
                    break;
                case 3:
                    print("STILL SITTING....");
                    u = 0;
                    break;
                case 4:
                    print("NEVER FLITTING");
                    u = 2;
                    break;
                case 5:
                    print("BURNED");
                    break;
            }
        } else if (j == 3) {
            switch (i) {
                case 1:
                    print("AND MY SOUL");
                    break;
                case 2:
                    print("DARKNESS THERE");
                    break;
                case 3:
                    print("SHALL BE LIFTED");
                    break;
                case 4:
                    print("QUOTH THE RAVEN");
                    break;
                case 5:
                    if (u == 0)
                        break;
                    print("SIGN OF PARTING");
                    break;
            }
        } else if (j == 4) {
            switch (i) {
                case 1:
                    print("NOTHING MORE");
                    break;
                case 2:
                    print("YET AGAIN");
                    break;
                case 3:
                    print("SLOWLY CREEPING");
                    break;
                case 4:
                    print("...EVERMORE");
                    break;
                case 5:
                    print("NEVERMORE");
                    break;
            }
        }
        // 根据条件随机打印逗号或空格，并更新 u 的值
        if (u != 0 && Math.random() <= 0.19) {
            print(",");
            u = 2;
        }
        if (Math.random() <= 0.65) {
            print(" ");
            u++;
        } else {
            print("\n");
            u = 0;
        }
        // 无限循环，生成新的 i 和 j 的值
        while (1) {
            i = Math.floor(Math.floor(10 * Math.random()) / 2) + 1;
            j++;
            k++;
            if (u == 0 && j % 2 == 0)
                print("     ");
            if (j != 5)
                break;
            j = 0;
            print("\n");
            if (k <= 20)
                continue;
            print("\n");
            u = 0;
            k = 0;
            j = 2;
            break;
        }
        // 根据条件判断是否结束循环
        if (u == 0 && k == 0 && j == 2 && ++times == 10)
            break;
    }
}

// 调用主程序
main();

```