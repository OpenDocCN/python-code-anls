# `.\PaddleOCR\deploy\fastdeploy\android\app\src\main\java\com\baidu\paddle\fastdeploy\app\ui\layout\ActionBarLayout.java`

```
package com.baidu.paddle.fastdeploy.app.ui.layout;

import android.content.Context;
import android.graphics.Color;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.widget.RelativeLayout;

// 自定义的 ActionBar 布局类，继承自 RelativeLayout
public class ActionBarLayout extends RelativeLayout {
    // 定义布局高度为 150
    private int layoutHeight = 150;

    // 构造函数，接受一个 Context 参数
    public ActionBarLayout(Context context) {
        super(context);
    }

    // 构造函数，接受一个 Context 和 AttributeSet 参数
    public ActionBarLayout(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    // 构造函数，接受一个 Context、AttributeSet 和 defStyleAttr 参数
    public ActionBarLayout(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    // 重写 onMeasure 方法，用于测量布局的宽高
    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        // 获取测量宽度
        int width = MeasureSpec.getSize(widthMeasureSpec);
        // 设置测量宽度和自定义的布局高度
        setMeasuredDimension(width, layoutHeight);
        // 设置背景颜色为黑色
        setBackgroundColor(Color.BLACK);
        // 设置透明度为 0.9
        setAlpha(0.9f);
    }
}
```