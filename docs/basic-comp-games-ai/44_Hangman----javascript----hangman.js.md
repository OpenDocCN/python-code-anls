# `basic-computer-games\44_Hangman\javascript\hangman.js`

```

// HANGMAN
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串输出到指定元素
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
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

// 定义制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 输出游戏标题
print(tab(32) + "HANGMAN\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");

// 初始化变量
var pa = [];
var la = [];
var da = [];
var na = [];
var ua = [];

// 定义单词列表
var words = ["GUM","SIN","FOR","CRY","LUG","BYE","FLY",
             "UGLY","EACH","FROM","WORK","TALK","WITH","SELF",
             "PIZZA","THING","FEIGN","FIEND","ELBOW","FAULT","DIRTY",
             "BUDGET","SPIRIT","QUAINT","MAIDEN","ESCORT","PICKAX",
             "EXAMPLE","TENSION","QUININE","KIDNEY","REPLICA","SLEEPER",
             "TRIANGLE","KANGAROO","MAHOGANY","SERGEANT","SEQUENCE",
             "MOUSTACHE","DANGEROUS","SCIENTIST","DIFFERENT","QUIESCENT",
             "MAGISTRATE","ERRONEOUSLY","LOUDSPEAKER","PHYTOTOXIC",
             "MATRIMONIAL","PARASYMPATHOMIMETIC","THIGMOTROPISM"];

// 主控制部分
async function main()
}

// 调用主函数
main();

```