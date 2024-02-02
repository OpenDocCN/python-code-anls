# `MetaGPT\tests\data\code\js\1.js`

```py

# 定义一个名为 WRMCB 的函数，用于处理批量脚本运行时的错误
WRMCB=function(e){var c=console;if(c&&c.log&&c.error){c.log('Error running batched script.');c.error(e);}}
;
# 尝试执行以下代码块，捕获可能出现的错误
try {
    /* module-key = 'jira.webresources:bigpipe-js', location = '/includes/jira/common/bigpipe.js' */
    # 定义一个模块，依赖于 jquery、wrm/data、jira/skate 和 jira/util/logger
    define("jira/bigpipe/element",["jquery","wrm/data","jira/skate","jira/util/logger"],function(e,r,t,n){
        # 返回一个名为 big-pipe 的自定义元素
        return t("big-pipe",{
            # 当元素被附加到文档时执行的函数
            attached:function(i){
                # 定义内部函数和变量
                function a(){var e=new CustomEvent("success");i.dispatchEvent(e)}
                function o(e,r){var t=new CustomEvent("error");t.data={event:e,signature:r},i.dispatchEvent(t)}
                function d(e,r){p("error"),o(e,r)}
                function p(e){"performance"in window&&performance.mark&&performance.mark(c+e)}
                # 获取元素的 data-id 属性值
                var s=i.getAttribute("data-id");
                # 如果 data-id 为 null，则记录错误并返回
                if(null===s)return n.error("No data-id attribute provided for tag <big-pipe/> for element:",i),void d({name:"NoPipeIdError",message:"Unable to render element. Element does not contain a pipe id.",element:i},"no.pipe.id");
                # 定义性能标记的名称前缀
                var c="bigPipe."+s+".";
                # 执行性能标记
                p("start");
                # 从数据仓库中获取与 data-id 对应的数据
                var u=r.claim(s);
                # 如果获取到数据，则执行以下操作
                u?function(r){
                    try{
                        # 解析数据并替换元素内容，然后初始化替换后的元素
                        var o=e(r);
                        e(i).replaceWith(o).each(function(){t.init(this)}),p("end"),a()
                    }catch(e){
                        # 记录解析错误并返回
                        n.error("Error while parsing html: "+e),d(e,"parsing")
                    }
                }(u):
                # 如果未获取到数据，则记录错误并返回
                d({name:"NoDataError",message:"BigPipe response is empty."},"no.data")
            },
            # 当元素从文档中移除时执行的函数
            detached:function(){},
            # 元素类型为自定义元素
            type:t.type.ELEMENT,
            # 已解析的属性名
            resolvedAttribute:"resolved",
            # 未解析的属性名
            unresolvedAttribute:"unresolved"
        })
    });
# 捕获可能出现的错误，并调用 WRMCB 函数处理
}catch(e){WRMCB(e)};

```