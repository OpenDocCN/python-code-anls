# `.\AutoGPT\benchmark\frontend\src\server\db.ts`

```py
# 导入 PrismaClient 类和 env 变量
import { PrismaClient } from "@prisma/client";
import { env } from "~/env.mjs";

# 将 globalThis 强制转换为未知类型的对象，并赋值给 globalForPrisma
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

# 导出 prisma 变量
export const prisma =
  # 如果 globalForPrisma 中已经存在 prisma 对象，则使用该对象；否则创建一个新的 PrismaClient 对象
  globalForPrisma.prisma ??
  new PrismaClient({
    # 根据环境变量设置日志级别
    log:
      env.NODE_ENV === "development" ? ["query", "error", "warn"] : ["error"],
  });

# 如果环境不是生产环境，则将 prisma 对象赋值给 globalForPrisma.prisma
if (env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
```