# `.\PaddleOCR\deploy\fastdeploy\android\app\src\main\java\com\baidu\paddle\fastdeploy\app\ui\Utils.java`

```py
// 定义一个名为 Utils 的类，包含一些常用的工具方法
public class Utils {
    // 定义一个 TAG 常量，用于在日志中标识该类
    private static final String TAG = Utils.class.getSimpleName();

    // 递归创建目录的方法
    public static void RecursiveCreateDirectories(String fileDir) {
        // 将传入的文件路径按照 "/" 分割成数组
        String[] fileDirs = fileDir.split("\\/");
        // 初始化顶层路径为空字符串
        String topPath = "";
        // 遍历文件路径数组
        for (int i = 0; i < fileDirs.length; i++) {
            // 拼接顶层路径
            topPath += "/" + fileDirs[i];
            // 创建一个 File 对象，表示当前路径
            File file = new File(topPath);
            // 如果当前路径已经存在，则继续下一个路径
            if (file.exists()) {
                continue;
            } else {
                // 如果当前路径不存在，则创建该路径
                file.mkdir();
            }
        }
    }
}
    // 从应用的 Assets 目录中复制文件到指定路径
    public static void copyFileFromAssets(Context appCtx, String srcPath, String dstPath) {
        // 如果源路径或目标路径为空，则直接返回
        if (srcPath.isEmpty() || dstPath.isEmpty()) {
            return;
        }
        // 获取目标路径的目录部分
        String dstDir = dstPath.substring(0, dstPath.lastIndexOf('/'));
        // 如果目标目录长度大于0，则递归创建目录
        if (dstDir.length() > 0) {
            RecursiveCreateDirectories(dstDir);
        }
        InputStream is = null;
        OutputStream os = null;
        try {
            // 从 Assets 中打开源文件的输入流
            is = new BufferedInputStream(appCtx.getAssets().open(srcPath));
            // 创建目标文件的输出流
            os = new BufferedOutputStream(new FileOutputStream(new File(dstPath)));
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
            try {
                // 关闭输入输出流
                os.close();
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // 从应用的 Assets 目录中复制整个目录到指定路径
    public static void copyDirectoryFromAssets(Context appCtx, String srcDir, String dstDir) {
        // 如果源目录或目标目录为空，则直接返回
        if (srcDir.isEmpty() || dstDir.isEmpty()) {
            return;
        }
        try {
            // 如果目标目录不存在，则递归创建目录
            if (!new File(dstDir).exists()) {
                new File(dstDir).mkdirs();
            }
            // 遍历源目录下的文件和子目录
            for (String fileName : appCtx.getAssets().list(srcDir)) {
                String srcSubPath = srcDir + File.separator + fileName;
                String dstSubPath = dstDir + File.separator + fileName;
                // 如果是子目录，则递归复制子目录
                if (new File(srcSubPath).isDirectory()) {
                    copyDirectoryFromAssets(appCtx, srcSubPath, dstSubPath);
                } else {
                    // 如果是文件，则调用复制文件方法
                    copyFileFromAssets(appCtx, srcSubPath, dstSubPath);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    # 从字符串中解析出浮点数数组
    public static float[] parseFloatsFromString(String string, String delimiter) {
        # 去除字符串两端空格并转换为小写，然后根据分隔符拆分成数组
        String[] pieces = string.trim().toLowerCase().split(delimiter);
        # 创建与 pieces 长度相同的浮点数数组
        float[] floats = new float[pieces.length];
        # 遍历 pieces 数组，将每个元素转换为浮点数并存入 floats 数组
        for (int i = 0; i < pieces.length; i++) {
            floats[i] = Float.parseFloat(pieces[i].trim());
        }
        # 返回浮点数数组
        return floats;
    }
    
    # 从字符串中解析出长整型数组
    public static long[] parseLongsFromString(String string, String delimiter) {
        # 去除字符串两端空格并转换为小写，然后根据分隔符拆分成数组
        String[] pieces = string.trim().toLowerCase().split(delimiter);
        # 创建与 pieces 长度相同的长整型数组
        long[] longs = new long[pieces.length];
        # 遍历 pieces 数组，将每个元素转换为长整型并存入 longs 数组
        for (int i = 0; i < pieces.length; i++) {
            longs[i] = Long.parseLong(pieces[i].trim());
        }
        # 返回长整型数组
        return longs;
    }
    
    # 获取SD卡目录路径
    public static String getSDCardDirectory() {
        # 返回外部存储目录的绝对路径
        return Environment.getExternalStorageDirectory().getAbsolutePath();
    }
    
    # 获取DCIM目录路径
    public static String getDCIMDirectory() {
        # 返回外部公共存储目录中的 DCIM 目录的绝对路径
        return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).getAbsolutePath();
    }
    // 获取最佳预览尺寸，根据传入的尺寸列表和宽高参数
    public static Camera.Size getOptimalPreviewSize(List<Camera.Size> sizes, int w, int h) {
        // 定义宽高比容差
        final double ASPECT_TOLERANCE = 0.3;
        // 计算目标宽高比
        double targetRatio = (double) w / h;
        // 如果尺寸列表为空，返回空
        if (sizes == null) return null;

        // 初始化最佳尺寸和最小差值
        Camera.Size optimalSize = null;
        double minDiff = Double.MAX_VALUE;

        int targetHeight = h;

        // 尝试找到符合宽高比和尺寸的尺寸
        for (Camera.Size size : sizes) {
            double ratio = (double) size.width / size.height;
            // 如果宽高比与目标宽高比的差值大于容差，继续下一次循环
            if (Math.abs(ratio - targetRatio) > ASPECT_TOLERANCE) continue;
            // 如果当前尺寸的高度与目标高度的差值小于最小差值，更新最佳尺寸和最小差值
            if (Math.abs(size.height - targetHeight) < minDiff) {
                optimalSize = size;
                minDiff = Math.abs(size.height - targetHeight);
            }
        }

        // 无法找到符合宽高比的尺寸，忽略要求
        if (optimalSize == null) {
            minDiff = Double.MAX_VALUE;
            for (Camera.Size size : sizes) {
                // 如果当前尺寸的高度与目标高度的差值小于最小差值，更新最佳尺寸和最小差值
                if (Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(size.height - targetHeight);
                }
            }
        }
        // 返回最佳尺寸
        return optimalSize;
    }

    // 获取屏幕宽度
    public static int getScreenWidth() {
        return Resources.getSystem().getDisplayMetrics().widthPixels;
    }

    // 获取屏幕高度
    public static int getScreenHeight() {
        return Resources.getSystem().getDisplayMetrics().heightPixels;
    }
    // 获取相机显示方向
    public static int getCameraDisplayOrientation(Context context, int cameraId) {
        // 创建相机信息对象
        Camera.CameraInfo info = new Camera.CameraInfo();
        // 获取相机信息
        Camera.getCameraInfo(cameraId, info);
        // 获取窗口管理器
        WindowManager wm = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        // 获取当前显示的旋转角度
        int rotation = wm.getDefaultDisplay().getRotation();
        // 初始化角度变量
        int degrees = 0;
        // 根据旋转角度设置角度值
        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
        }
        // 初始化结果变量
        int result;
        // 如果是前置摄像头
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            // 计算结果角度
            result = (info.orientation + degrees) % 360;
            // 补偿镜像效果
            result = (360 - result) % 360;   // compensate the mirror
        } else {
            // 后置摄像头
            // 计算结果角度
            result = (info.orientation - degrees + 360) % 360;
        }
        // 返回最终结果
        return result;
    }
    // 创建着色器程序，接收顶点着色器代码和片段着色器代码作为参数
    public static int createShaderProgram(String vss, String fss) {
        // 创建顶点着色器对象
        int vshader = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);
        // 将顶点着色器代码加载到着色器对象中
        GLES20.glShaderSource(vshader, vss);
        // 编译顶点着色器
        GLES20.glCompileShader(vshader);
        // 获取编译状态
        int[] status = new int[1];
        GLES20.glGetShaderiv(vshader, GLES20.GL_COMPILE_STATUS, status, 0);
        // 如果编译失败，则输出错误信息，删除着色器对象，返回0
        if (status[0] == 0) {
            Log.e(TAG, GLES20.glGetShaderInfoLog(vshader));
            GLES20.glDeleteShader(vshader);
            vshader = 0;
            return 0;
        }

        // 创建片段着色器对象
        int fshader = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
        // 将片段着色器代码加载到着色器对象中
        GLES20.glShaderSource(fshader, fss);
        // 编译片段着色器
        GLES20.glCompileShader(fshader);
        // 获取编译状态
        GLES20.glGetShaderiv(fshader, GLES20.GL_COMPILE_STATUS, status, 0);
        // 如果编译失败，则输出错误信息，删除着色器对象，返回0
        if (status[0] == 0) {
            Log.e(TAG, GLES20.glGetShaderInfoLog(fshader));
            GLES20.glDeleteShader(vshader);
            GLES20.glDeleteShader(fshader);
            fshader = 0;
            return 0;
        }

        // 创建着色器程序对象
        int program = GLES20.glCreateProgram();
        // 将顶点着色器和片段着色器附加到着色器程序对象上
        GLES20.glAttachShader(program, vshader);
        GLES20.glAttachShader(program, fshader);
        // 链接着色器程序
        GLES20.glLinkProgram(program);
        // 删除顶点着色器和片段着色器对象
        GLES20.glDeleteShader(vshader);
        GLES20.glDeleteShader(fshader);
        // 获取链接状态
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, status, 0);
        // 如果链接失败，则输出错误信息，删除着色器程序对象，返回0
        if (status[0] == 0) {
            Log.e(TAG, GLES20.glGetProgramInfoLog(program));
            program = 0;
            return 0;
        }
        // 验证着色器程序
        GLES20.glValidateProgram(program);
        // 获取验证状态
        GLES20.glGetProgramiv(program, GLES20.GL_VALIDATE_STATUS, status, 0);
        // 如果验证失败，则输出错误信息，删除着色器程序对象，返回0
        if (status[0] == 0) {
            Log.e(TAG, GLES20.glGetProgramInfoLog(program));
            GLES20.glDeleteProgram(program);
            program = 0;
            return 0;
        }

        // 返回着色器程序对象
        return program;
    }
    // 检查设备是否支持 NPU，判断硬件是否为 "kirin810" 或 "kirin990"
    public static boolean isSupportedNPU() {
        // 获取设备硬件信息
        String hardware = android.os.Build.HARDWARE;
        // 判断硬件是否为 "kirin810" 或 "kirin990"
        return hardware.equalsIgnoreCase("kirin810") || hardware.equalsIgnoreCase("kirin990");
    }

    // 解码指定路径的图片文件，并按指定的显示宽度和高度进行缩放
    public static Bitmap decodeBitmap(String path, int displayWidth, int displayHeight) {
        // 创建 BitmapFactory.Options 对象
        BitmapFactory.Options op = new BitmapFactory.Options();
        // 仅读取 Bitmap 的宽度和高度信息，不读取像素信息
        op.inJustDecodeBounds = true;
        // 读取指定路径的图片文件，获取大小信息
        Bitmap bmp = BitmapFactory.decodeFile(path, op);
        // 计算宽度和高度的缩放比例
        int wRatio = (int) Math.ceil(op.outWidth / (float) displayWidth);
        int hRatio = (int) Math.ceil(op.outHeight / (float) displayHeight);
        // 如果超过指定大小，减小相应的缩放比例
        if (wRatio > 1 && hRatio > 1) {
            if (wRatio > hRatio) {
                // 如果宽度过大，将宽度缩小到所需大小，注意高度也会变小
                op.inSampleSize = wRatio;
            } else {
                op.inSampleSize = hRatio;
            }
        }
        // 重新设置 inJustDecodeBounds 为 false
        op.inJustDecodeBounds = false;
        // 重新读取指定路径的图片文件，应用缩放比例
        bmp = BitmapFactory.decodeFile(path, op);
        // 从原始 Bitmap 创建具有给定宽度和高度的 Bitmap
        return Bitmap.createScaledBitmap(bmp, displayWidth, displayHeight, true);
    }
    // 从内容 URI 获取真实路径
    public static String getRealPathFromURI(Context context, Uri contentURI) {
        String result;
        Cursor cursor = null;
        try {
            // 查询内容 URI 获取游标
            cursor = context.getContentResolver().query(contentURI, null, null, null, null);
        } catch (Throwable e) {
            // 捕获异常并打印堆栈信息
            e.printStackTrace();
        }
        // 如果游标为空，则使用内容 URI 的路径作为结果
        if (cursor == null) {
            result = contentURI.getPath();
        } else {
            // 移动游标到第一行
            cursor.moveToFirst();
            // 获取图片数据列的索引
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            // 获取真实路径
            result = cursor.getString(idx);
            // 关闭游标
            cursor.close();
        }
        // 返回结果路径
        return result;
    }

    // 读取文本文件内容并返回为字符串列表
    public static List<String> readTxt(String txtPath) {
        // 创建文件对象
        File file = new File(txtPath);
        // 检查文件是否存在且为文件
        if (file.isFile() && file.exists()) {
            try {
                // 创建文件输入流
                FileInputStream fileInputStream = new FileInputStream(file);
                // 创建输入流读取器
                InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream);
                // 创建缓冲读取器
                BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
                String text;
                // 创建字符串列表用于存储每行文本
                List<String> labels = new ArrayList<>();
                // 逐行读取文本并添加到列表中
                while ((text = bufferedReader.readLine()) != null) {
                    labels.add(text);
                }
                // 返回文本行列表
                return labels;
            } catch (Exception e) {
                // 捕获异常并打印堆栈信息
                e.printStackTrace();
            }
        }
        // 如果文件不存在或不是文件，则返回空
        return null;
    }
# 闭合之前的代码块
```