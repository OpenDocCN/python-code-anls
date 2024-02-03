# `.\PaddleOCR\deploy\fastdeploy\android\app\src\main\java\com\baidu\paddle\fastdeploy\app\examples\ocr\OcrSettingsActivity.java`

```py
package com.baidu.paddle.fastdeploy.app.examples.ocr;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.EditTextPreference;
import android.preference.ListPreference;
importandroid.preference.PreferenceManager;
import android.support.v7.app.ActionBar;

import com.baidu.paddle.fastdeploy.app.examples.R;
import com.baidu.paddle.fastdeploy.app.ui.Utils;
import com.baidu.paddle.fastdeploy.app.ui.view.AppCompatPreferenceActivity;

import java.util.ArrayList;
import java.util.List;

public class OcrSettingsActivity extends AppCompatPreferenceActivity implements
        SharedPreferences.OnSharedPreferenceChangeListener {
    private static final String TAG = OcrSettingsActivity.class.getSimpleName();

    static public int selectedModelIdx = -1;
    static public String modelDir = "";
    static public String labelPath = "";
    static public int cpuThreadNum = 2;
    static public String cpuPowerMode = "";
    static public float scoreThreshold = 0.4f;
    static public String enableLiteFp16 = "true";

    ListPreference lpChoosePreInstalledModel = null;
    EditTextPreference etModelDir = null;
    EditTextPreference etLabelPath = null;
    ListPreference lpCPUThreadNum = null;
    ListPreference lpCPUPowerMode = null;
    EditTextPreference etScoreThreshold = null;
    ListPreference lpEnableLiteFp16 = null;

    List<String> preInstalledModelDirs = null;
    List<String> preInstalledLabelPaths = null;
    List<String> preInstalledCPUThreadNums = null;
    List<String> preInstalledCPUPowerModes = null;
    List<String> preInstalledScoreThresholds = null;
    List<String> preInstalledEnableLiteFp16s = null;

    // 重写父类方法，处理设置界面的变化
    @Override
    }

    // 标记该方法会忽略特定的警告
    @SuppressLint("ApplySharedPref")
    }
    // 检查并更新应用程序的设置
    static boolean checkAndUpdateSettings(Context ctx) {
        // 标记设置是否有更改
        boolean settingsChanged = false;
        // 获取默认的共享偏好设置
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(ctx);

        // 获取模型目录设置
        String model_dir = sharedPreferences.getString(ctx.getString(R.string.MODEL_DIR_KEY),
                ctx.getString(R.string.OCR_MODEL_DIR_DEFAULT));
        // 检查模型目录设置是否有更改
        settingsChanged |= !modelDir.equalsIgnoreCase(model_dir);
        // 更新模型目录设置
        modelDir = model_dir;

        // 获取标签路径设置
        String label_path = sharedPreferences.getString(ctx.getString(R.string.LABEL_PATH_KEY),
                ctx.getString(R.string.OCR_REC_LABEL_DEFAULT));
        // 检查标签路径设置是否有更改
        settingsChanged |= !labelPath.equalsIgnoreCase(label_path);
        // 更新标签路径设置
        labelPath = label_path;

        // 获取 CPU 线程数设置
        String cpu_thread_num = sharedPreferences.getString(ctx.getString(R.string.CPU_THREAD_NUM_KEY),
                ctx.getString(R.string.CPU_THREAD_NUM_DEFAULT));
        // 检查 CPU 线程数设置是否有更改
        settingsChanged |= cpuThreadNum != Integer.parseInt(cpu_thread_num);
        // 更新 CPU 线程数设置
        cpuThreadNum = Integer.parseInt(cpu_thread_num);

        // 获取 CPU 功耗模式设置
        String cpu_power_mode = sharedPreferences.getString(ctx.getString(R.string.CPU_POWER_MODE_KEY),
                ctx.getString(R.string.CPU_POWER_MODE_DEFAULT));
        // 检查 CPU 功耗模式设置是否有更改
        settingsChanged |= !cpuPowerMode.equalsIgnoreCase(cpu_power_mode);
        // 更新 CPU 功耗模式设置
        cpuPowerMode = cpu_power_mode;

        // 获取分数阈值设置
        String score_threshold = sharedPreferences.getString(ctx.getString(R.string.SCORE_THRESHOLD_KEY),
                ctx.getString(R.string.SCORE_THRESHOLD_DEFAULT));
        // 检查分数阈值设置是否有更改
        settingsChanged |= scoreThreshold != Float.parseFloat(score_threshold);
        // 更新分数阈值设置
        scoreThreshold = Float.parseFloat(score_threshold);

        // 获取启用 Lite FP16 模式设置
        String enable_lite_fp16 = sharedPreferences.getString(ctx.getString(R.string.ENABLE_LITE_FP16_MODE_KEY),
                ctx.getString(R.string.ENABLE_LITE_FP16_MODE_DEFAULT));
        // 检查启用 Lite FP16 模式设置是否有更改
        settingsChanged |= !enableLiteFp16.equalsIgnoreCase(enable_lite_fp16);
        // 更新启用 Lite FP16 模式设置
        enableLiteFp16 = enable_lite_fp16;

        // 返回设置是否有更改的结果
        return settingsChanged;
    }
    // 重置设置为默认值
    static void resetSettings() {
        selectedModelIdx = -1;
        modelDir = "";
        labelPath = "";
        cpuThreadNum = 2;
        cpuPowerMode = "";
        scoreThreshold = 0.4f;
        enableLiteFp16 = "true";
    }

    // 当 Activity 重新进入前台时调用
    @Override
    protected void onResume() {
        super.onResume();
        // 注册 SharedPreferences 的变化监听器
        getPreferenceScreen().getSharedPreferences().registerOnSharedPreferenceChangeListener(this);
        // 重新加载设置并更新 UI
        reloadSettingsAndUpdateUI();
    }

    // 当 Activity 进入后台时调用
    @Override
    protected void onPause() {
        super.onPause();
        // 取消注册 SharedPreferences 的变化监听器
        getPreferenceScreen().getSharedPreferences().unregisterOnSharedPreferenceChangeListener(this);
    }

    // 当 SharedPreferences 中的设置发生变化时调用
    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        // 重新加载设置并更新 UI
        reloadSettingsAndUpdateUI();
    }
# 闭合之前的代码块
```