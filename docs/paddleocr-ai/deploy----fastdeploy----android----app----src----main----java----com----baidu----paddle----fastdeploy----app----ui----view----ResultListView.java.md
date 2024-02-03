# `.\PaddleOCR\deploy\fastdeploy\android\app\src\main\java\com\baidu\paddle\fastdeploy\app\ui\view\ResultListView.java`

```
package com.baidu.paddle.fastdeploy.app.ui.view;

import android.content.Context;
import android.os.Handler;
import android.util.AttributeSet;
import android.widget.ListView;

public class ResultListView extends ListView {
    // 构造函数，接收上下文参数
    public ResultListView(Context context) {
        super(context);
    }

    // 构造函数，接收上下文和属性参数
    public ResultListView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    // 构造函数，接收上下文、属性和样式参数
    public ResultListView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    // 声明 Handler 对象
    private Handler handler;

    // 设置 Handler 对象
    public void setHandler(Handler mHandler) {
        handler = mHandler;
    }

    // 清空 ListView 中的内容
    public void clear() {
        // 在 Handler 中执行清空操作
        handler.post(new Runnable() {
            @Override
            public void run() {
                // 移除所有子视图
                removeAllViewsInLayout();
                // 使视图无效，触发重绘
                invalidate();
            }
        });
    }

    // 重写测量方法
    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        // 创建一个扩展的测量规格，使 ListView 可以展开
        int expandSpec = MeasureSpec.makeMeasureSpec(Integer.MAX_VALUE >> 2,
                MeasureSpec.AT_MOST);
        // 调用父类的测量方法
        super.onMeasure(widthMeasureSpec, expandSpec);
    }
}
```