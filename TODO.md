# Pytorch Operation

1. 在使用 torch 的时候，通常不会对数据创建副本，但是 在处理非连续的内存时，执行`contiguous`、`reshape`这样的操作会创建数据副本，在数据副本中是内存是连续的

2. 如果两个矩阵做矩阵乘法，浮点运算次数就等于三个维度的乘积的两倍

例子:`x:(B,D) y:(D,k) -> count_num_flops = 2 * B * D * K`

其中 B 代表数据点的数量，DK 代表参数数量，这样一个模型的前向传播的浮点运算量**大约**是 token 数的两倍

前向传播的计算量约为参数量的 2 倍，反向传播约为参数量的 4 倍

## MFU: Model FLOPs utilization

Definition：(actual FLOP/s)/(promised FLOP/s) [ignore communication/overhead]
