# `.\PaddleOCR\deploy\fastdeploy\android\app\src\main\java\com\baidu\paddle\fastdeploy\app\ui\view\adapter\BaseResultAdapter.java`

```
package com.baidu.paddle.fastdeploy.app.ui.view.adapter;

import android.content.Context;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import com.baidu.paddle.fastdeploy.app.examples.R;
import com.baidu.paddle.fastdeploy.app.ui.view.model.BaseResultModel;

import java.text.DecimalFormat;
import java.util.List;

public class BaseResultAdapter extends ArrayAdapter<BaseResultModel> {
    private int resourceId;

    public BaseResultAdapter(@NonNull Context context, int resource) {
        super(context, resource);
    }

    public BaseResultAdapter(@NonNull Context context, int resource, @NonNull List<BaseResultModel> objects) {
        super(context, resource, objects);
        resourceId = resource;
    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        // 获取当前位置的 BaseResultModel 对象
        BaseResultModel model = getItem(position);
        // 根据布局资源文件创建视图
        View view = LayoutInflater.from(getContext()).inflate(resourceId, null);
        // 获取视图中的 TextView 控件
        TextView indexText = (TextView) view.findViewById(R.id.index);
        TextView nameText = (TextView) view.findViewById(R.id.name);
        TextView confidenceText = (TextView) view.findViewById(R.id.confidence);
        // 设置 TextView 控件的文本内容
        indexText.setText(String.valueOf(model.getIndex()));
        nameText.setText(String.valueOf(model.getName()));
        confidenceText.setText(formatFloatString(model.getConfidence()));
        // 返回填充好数据的视图
        return view;
    }

    // 格式化浮点数为字符串，保留两位小数
    public static String formatFloatString(float number) {
        DecimalFormat df = new DecimalFormat("0.00");
        return df.format(number);
    }
}
```