# `.\PaddleOCR\deploy\android_demo\app\src\main\java\com\baidu\paddle\lite\demo\ocr\Utils.java`

```py
// 导入必要的类
package com.baidu.paddle.lite.demo.ocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.os.Environment;

import java.io.*;

// 创建 Utils 类
public class Utils {
    // 定义 TAG 常量
    private static final String TAG = Utils.class.getSimpleName();

    // 从 Assets 目录复制文件到指定路径
    public static void copyFileFromAssets(Context appCtx, String srcPath, String dstPath) {
        // 如果源路径或目标路径为空，则直接返回
        if (srcPath.isEmpty() || dstPath.isEmpty()) {
            return;
        }
        // 初始化输入流和输出流
        InputStream is = null;
        OutputStream os = null;
        try {
            // 从 Assets 目录打开源文件输入流
            is = new BufferedInputStream(appCtx.getAssets().open(srcPath));
            // 创建目标文件输出流
            os = new BufferedOutputStream(new FileOutputStream(new File(dstPath)));
            // 创建缓冲区和长度变量
            byte[] buffer = new byte[1024];
            int length = 0;
            // 读取源文件内容并写入目标文件
            while ((length = is.read(buffer)) != -1) {
                os.write(buffer, 0, length);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            // 关闭输入流和输出流
            try {
                os.close();
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
    // 从 assets 目录中复制整个文件夹到指定目录
    public static void copyDirectoryFromAssets(Context appCtx, String srcDir, String dstDir) {
        // 如果源目录或目标目录为空，则直接返回
        if (srcDir.isEmpty() || dstDir.isEmpty()) {
            return;
        }
        try {
            // 如果目标目录不存在，则创建目标目录
            if (!new File(dstDir).exists()) {
                new File(dstDir).mkdirs();
            }
            // 遍历源目录下的文件和文件夹
            for (String fileName : appCtx.getAssets().list(srcDir)) {
                // 构建源子路径和目标子路径
                String srcSubPath = srcDir + File.separator + fileName;
                String dstSubPath = dstDir + File.separator + fileName;
                // 如果是文件夹，则递归复制文件夹
                if (new File(srcSubPath).isDirectory()) {
                    copyDirectoryFromAssets(appCtx, srcSubPath, dstSubPath);
                } else {
                    // 如果是文件，则复制文件
                    copyFileFromAssets(appCtx, srcSubPath, dstSubPath);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 从字符串中解析出浮点数数组
    public static float[] parseFloatsFromString(String string, String delimiter) {
        // 去除字符串两端空格并转换为小写，然后按分隔符拆分
        String[] pieces = string.trim().toLowerCase().split(delimiter);
        float[] floats = new float[pieces.length];
        // 将拆分后的字符串转换为浮点数数组
        for (int i = 0; i < pieces.length; i++) {
            floats[i] = Float.parseFloat(pieces[i].trim());
        }
        return floats;
    }

    // 从字符串中解析出长整型数组
    public static long[] parseLongsFromString(String string, String delimiter) {
        // 去除字符串两端空格并转换为小写，然后按分隔符拆分
        String[] pieces = string.trim().toLowerCase().split(delimiter);
        long[] longs = new long[pieces.length];
        // 将拆分后的字符串转换为长整型数组
        for (int i = 0; i < pieces.length; i++) {
            longs[i] = Long.parseLong(pieces[i].trim());
        }
        return longs;
    }

    // 获取SD卡目录路径
    public static String getSDCardDirectory() {
        return Environment.getExternalStorageDirectory().getAbsolutePath();
    }

    // 检查是否支持 NPU
    public static boolean isSupportedNPU() {
        return false;
        // String hardware = android.os.Build.HARDWARE;
        // return hardware.equalsIgnoreCase("kirin810") || hardware.equalsIgnoreCase("kirin990");
    }
    // 根据指定的最大长度和步长来调整图片大小
    public static Bitmap resizeWithStep(Bitmap bitmap, int maxLength, int step) {
        // 获取原始图片的宽度和高度
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        // 计算宽高中的最大值
        int maxWH = Math.max(width, height);
        // 初始化缩放比例、新宽度和新高度
        float ratio = 1;
        int newWidth = width;
        int newHeight = height;
        // 如果最大宽高超过指定最大长度，则按比例缩放
        if (maxWH > maxLength) {
            ratio = maxLength * 1.0f / maxWH;
            newWidth = (int) Math.floor(ratio * width);
            newHeight = (int) Math.floor(ratio * height);
        }

        // 根据步长调整新宽度
        newWidth = newWidth - newWidth % step;
        if (newWidth == 0) {
            newWidth = step;
        }
        // 根据步长调整新高度
        newHeight = newHeight - newHeight % step;
        if (newHeight == 0) {
            newHeight = step;
        }
        // 返回调整后的图片
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
    }
    // 旋转 Bitmap 图像
    public static Bitmap rotateBitmap(Bitmap bitmap, int orientation) {
        // 创建一个矩阵对象
        Matrix matrix = new Matrix();
        // 根据方向进行不同的旋转操作
        switch (orientation) {
            case ExifInterface.ORIENTATION_NORMAL:
                // 如果方向是正常的，直接返回原始图像
                return bitmap;
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                // 水平翻转
                matrix.setScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                // 旋转180度
                matrix.setRotate(180);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                // 垂直翻转
                matrix.setRotate(180);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                // 转置
                matrix.setRotate(90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                // 旋转90度
                matrix.setRotate(90);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                // 转置并旋转
                matrix.setRotate(-90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                // 旋转270度
                matrix.setRotate(-90);
                break;
            default:
                // 默认情况下返回原始图像
                return bitmap;
        }
        try {
            // 根据矩阵创建旋转后的 Bitmap 对象
            Bitmap bmRotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            // 回收原始 Bitmap 对象
            bitmap.recycle();
            // 返回旋转后的 Bitmap 对象
            return bmRotated;
        }
        catch (OutOfMemoryError e) {
            // 捕获内存溢出异常
            e.printStackTrace();
            // 返回空对象
            return null;
        }
    }
# 闭合之前的代码块
```