# `.\pytorch\third_party\miniz-2.1.0\examples\example6.c`

```
// example6.c - Demonstrates how to use miniz's PNG writer function.
// Public domain, April 11 2012, Rich Geldreich, richgel99@gmail.com.
// See "unlicense" statement at the end of tinfl.c.
// Mandlebrot set code from http://rosettacode.org/wiki/Mandelbrot_set#C
// Must link this example against libm on Linux.

// Purposely disable a whole bunch of stuff this low-level example doesn't use.
#define MINIZ_NO_STDIO       // Exclude standard IO functions from miniz
#define MINIZ_NO_TIME        // Exclude time-related functions from miniz
#define MINIZ_NO_ZLIB_APIS   // Exclude zlib APIs from miniz
#include "miniz.h"           // Include miniz header file for PNG writing functionality

#include <stdio.h>           // Include standard IO functions for file handling
#include <limits.h>          // Include limits for INT_MAX and INT_MIN
#include <math.h>            // Include math library for mathematical operations

typedef unsigned char uint8; // Define uint8 as unsigned char
typedef unsigned short uint16; // Define uint16 as unsigned short
typedef unsigned int uint;   // Define uint as unsigned int

typedef struct               // Define a structure for RGB color representation
{
  uint8 r, g, b;            // Red, Green, Blue components
} rgb_t;

static void hsv_to_rgb(int hue, int min, int max, rgb_t *p)
{
  const int invert = 0;     // Flag to invert hue
  const int saturation = 1; // Flag for saturation
  const int color_rotate = 0; // Color rotation

  if (min == max) max = min + 1; // Ensure max is not equal to min
  if (invert) hue = max - (hue - min); // Adjust hue if inverted
  if (!saturation) {
    p->r = p->g = p->b = 255 * (max - hue) / (max - min); // Convert to grayscale if no saturation
    return;
  }
  double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6); // Calculate hue
  double c = 255.0f * saturation; // Calculate chroma
  double X = c * (1 - fabs(fmod(h, 2) - 1)); // Calculate intermediate value

  p->r = p->g = p->b = 0;    // Initialize RGB components to 0

  switch((int)h) {           // Switch based on hue value
  case 0: p->r = c; p->g = X; return; // Red dominant
  case 1:    p->r = X; p->g = c; return; // Yellow dominant
  case 2: p->g = c; p->b = X; return; // Green dominant
  case 3: p->g = X; p->b = c; return; // Cyan dominant
  case 4: p->r = X; p->b = c; return; // Blue dominant
  default:p->r = c; p->b = X; // Magenta dominant
  }
}

int main(int argc, char *argv[])
{
  (void)argc, (void)argv;   // Suppress unused parameter warnings

  // Image resolution
  const int iXmax = 4096;   // Maximum X dimension of the image
  const int iYmax = 4096;   // Maximum Y dimension of the image

  // Output filename
  static const char *pFilename = "mandelbrot.png"; // Name of the PNG file to be generated

  int iX, iY;                // Iteration variables for X and Y coordinates
  const double CxMin = -2.5; // Minimum real coordinate of the initial area
  const double CxMax = 1.5;  // Maximum real coordinate of the initial area
  const double CyMin = -2.0; // Minimum imaginary coordinate of the initial area
  const double CyMax = 2.0;  // Maximum imaginary coordinate of the initial area

  double PixelWidth = (CxMax - CxMin) / iXmax; // Width of each pixel in the complex plane
  double PixelHeight = (CyMax - CyMin) / iYmax; // Height of each pixel in the complex plane

  // Z=Zx+Zy*i  ;   Z0 = 0
  double Zx, Zy;            // Real and imaginary parts of Z
  double Zx2, Zy2;          // Squares of Zx and Zy: Zx^2 and Zy^2

  int Iteration;            // Iteration count
  const int IterationMax = 200; // Maximum number of iterations for each point

  // Bail-out value, radius of circle
  const double EscapeRadius = 2; // Escape radius from the origin
  double ER2=EscapeRadius * EscapeRadius; // Square of escape radius

  uint8 *pImage = (uint8 *)malloc(iXmax * 3 * iYmax); // Allocate memory for image data

  // World (double) coordinate = parameter plane
  double Cx,Cy;             // Real and imaginary parts of C

  int MinIter = 9999, MaxIter = 0; // Minimum and maximum iterations for coloring

  for(iY = 0; iY < iYmax; iY++) // Iterate over Y axis of the image
  {
    Cy = CyMin + iY * PixelHeight; // Calculate current imaginary part of C
    if (fabs(Cy) < PixelHeight/2)
      Cy = 0.0; // Main antenna: set Cy to 0 if it's very close to the origin

    for(iX = 0; iX < iXmax; iX++) // Iterate over X axis of the image
    {
      // 计算当前像素点的颜色数组起始位置
      uint8 *color = pImage + (iX * 3) + (iY * iXmax * 3);
    
      // 计算复数C的实部Cx，根据当前像素点的位置iX和PixelWidth
      Cx = CxMin + iX * PixelWidth;
    
      // 设置初始值：轨道的初值为临界点Z=0
      Zx = 0.0;
      Zy = 0.0;
      Zx2 = Zx * Zx;
      Zy2 = Zy * Zy;
    
      // 迭代计算轨道的值，直到达到最大迭代次数或轨道逃逸半径的平方ER2
      for (Iteration = 0; Iteration < IterationMax && ((Zx2 + Zy2) < ER2); Iteration++)
      {
        Zy = 2 * Zx * Zy + Cy;
        Zx = Zx2 - Zy2 + Cx;
        Zx2 = Zx * Zx;
        Zy2 = Zy * Zy;
      };
    
      // 将迭代次数作为像素点的颜色值存入color数组
      color[0] = (uint8)Iteration;
      color[1] = (uint8)Iteration >> 8;
      color[2] = 0;
    
      // 更新最小迭代次数和最大迭代次数
      if (Iteration < MinIter)
        MinIter = Iteration;
      if (Iteration > MaxIter)
        MaxIter = Iteration;
    }
    
    for (iY = 0; iY < iYmax; iY++)
    {
      for (iX = 0; iX < iXmax; iX++)
      {
        // 获取当前像素点的颜色数组起始位置
        uint8 *color = (uint8 *)(pImage + (iX * 3) + (iY * iXmax * 3));
    
        // 从颜色数组中获取迭代次数
        uint Iterations = color[0] | (color[1] << 8U);
    
        // 将迭代次数转换为HSV色彩，并存入颜色数组中
        hsv_to_rgb(Iterations, MinIter, MaxIter, (rgb_t *)color);
      }
    }
    
    // 将图像数据写入PNG文件
    {
      // 计算PNG数据的大小
      size_t png_data_size = 0;
    
      // 使用tdefl库将图像数据写入内存中的PNG文件
      void *pPNG_data = tdefl_write_image_to_png_file_in_memory_ex(pImage, iXmax, iYmax, 3, &png_data_size, 6, MZ_FALSE);
      
      // 检查PNG数据是否写入成功
      if (!pPNG_data)
        fprintf(stderr, "tdefl_write_image_to_png_file_in_memory_ex() failed!\n");
      else
      {
        // 打开PNG文件以二进制写入模式
        FILE *pFile = fopen(pFilename, "wb");
    
        // 将PNG数据写入文件
        fwrite(pPNG_data, 1, png_data_size, pFile);
    
        // 关闭文件
        fclose(pFile);
    
        // 打印成功写入的文件名
        printf("Wrote %s\n", pFilename);
      }
    
      // 释放PNG数据内存
      mz_free(pPNG_data);
    }
    
    // 释放图像数据内存
    free(pImage);
    
    // 返回程序退出状态成功
    return EXIT_SUCCESS;
    }
}


注释：


# 这是一个代码块的结束，闭合了之前的函数或循环或条件语句等结构
```