实现一个高效的 RMSNorm 算子，它的核心是计算输入的均方根，

对应公式为： $\frac{1}{n} \sum_{i=1}^{n} x_i^2$。

我们需要分两个阶段进行归约计算：首先是warp间的归约，其次是warp内的归约来得到最终的结果。此外，当RMSNorm的输入包含多个样本时，还需启动多个线程块以实现各样本间的并行、独立计算，从而提升整体计算效率。

我们需要对每个输入样本在 batch 维度上独立进行归一化计算，其中每个样本的特征维度为 `dim`。这也与前文公式中的描述一致：输入为 `dim` 维的向量。我们在实现中，<mark>每个线程块将对应一个输入样本，独立完成其在 `dim` 维度上的归一化计算，从而实现批量数据的高效并行处理。</mark>

1. ###### 划分输入

在CUDA核函数中，每个线程块负责一个输入样本的RMSNorm计算，样本特征维度为`dim`（即 `size`）。通过`blockIdx.x` 获取线程块索引`bid`后，其对应的输入和输出数据起始地址为：`block_in`和`block_out`。每个线程块据此独立处理对应样本的数据。

```C++
__global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out,
                                    int batch, int size, float eps) {  
  const int bid = blockIdx.x;  
  if (bid >= batch) return;
  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  float sum = 0.0f;
```

2. ###### 归约

每个线程负责累加其在所属线程块内对应位置的数据。当线程块大小为 1024 时，线程0处理索引为 0、1024、2048、… 的元素，线程 1 处理 1、1025、2049、…，依此类推，线程 1023 处理 1023、2047、…。**每个线程累加其访问元素的平方值。随后，通过** **`block_reduce`** 对所有线程的局部累加结果进行归约。

该过程首先将 1024 个线程划分为 32 个 warp（每 warp 32 线程），在每个 warp 内部使用 warp-level 归约操作求出局部和，得到 32 个中间结果，为后续进一步归约提供输入。

```C++
for (int i = threadIdx.x; i < size; i += blockDim.x) {
   float x = block_in[i];
   sum += x * x;
}
__shared__ float shared_val;
sum = block_reduce(sum);
```

3. ###### BlockReduce的实现

![](https://tvle9mq8jh.feishu.cn/space/api/box/stream/download/asynccode/?code=N2Y3MjMyNDFmZGYxNTM2MjY1YmIxMzJlNzIwYjEwNDhfMGhrYkJYMUp6WlNWcmR0amNvVlZXU3VwcTVVSE1hNFBfVG9rZW46REh4ZWJYeVRxb2tGVWx4Nm5ZRGM0TXkxbkpoXzE3NTQ4MzY5NzE6MTc1NDg0MDU3MV9WNA)

我们先来回顾一下**`__shfl_down_sync`**函数，该函数用于在 warp 内实现**向下偏移（downward shift）**的数据交换。换句话说，lane ID 为 `t` 的线程将从 lane ID 为 `t + delta` 的线程中读取变量 `val` 的值，下图中的`delta`等于2；若 `t + delta >= width`，则保留当前线程自身的原始值 `val`。

![](https://tvle9mq8jh.feishu.cn/space/api/box/stream/download/asynccode/?code=NWNmOWNmMTFiMzhhYTQwYzljMTNlNzVlZjFjMjA1NWJfSzRzcTNEb2t6MDl2WklhUWpQNGw4SnA0VWFMWTVRY0lfVG9rZW46RWRVT2JHM1hFb1IyVTN4SjBLVmNiUkl5bnNjXzE3NTQ4MzY5NzE6MTc1NDg0MDU3MV9WNA)

图中也不难看出，thread 0从thread 2中获取对应的值，thread 1从thread 3中获取对应的值，直到线程 t，t + delta > 32，那么该线程就直接保留线程自身的值。<mark>基于上述思路，我们可以编写出一个高效的 `warpReduce` 函数，用于在单个线程束内对展开的32 个线程的数据进行归约求和。</mark>

理解上述原理后，掌握 `block_reduce` 的实现机制也就不再困难。其核心思想是通过多轮循环逐步归约，将一个线程块内所有线程的局部结果汇总到一个线程（通常是线程 0）。具体而言，在每一轮中，每个活跃线程将其当前累加值与另一个偏移 `stride` 的线程上的值进行合并。以 32 线程 warp 的归约为例：

- **第一轮（stride = 16）**：
  线程 0 与线程 16 的值相加，线程 1 与线程 17 相加，……，线程 15 与线程 31 相加。此后，前 16 个线程保存了前 32 个线程的成对部分和。

- **第二轮（stride = 8）**：
  线程 0 将当前值（已含线程 0 和 16 的和）与线程 8 的值相加（而线程 8 在上一轮已累加了线程 8 和 24 的和），于是线程 0 现在包含线程 0、8、16、24 的累加结果。同理，其他线程继续归约。

- 以此类推，`stride` 每轮减半（8 → 4 → 2 → 1），经过 5 轮后，线程 0 就聚合了整个 warp（线程 0 到 31）的所有原始数据之和。

```C++
__inline__ __device__ float block_reduce(float val) {
  const int tid = threadIdx.x;
  const int warpSize = 32;
  int lane = tid % warpSize;
  int warp_id = tid / warpSize;
  // Warp-level reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  // Write warp result to shared memory
  __shared__ float warpSums[32];
  if (lane == 0) {
    warpSums[warp_id] = val;
  }
```

我们这里32个warp经过各自独立的归约，都会得到一个局部的和，也就是有32个局部结果，我们将他们放在一个共享显存数组`warpSum`数组当中，以便进行下一个归约。将这这32个局部和再次累加，得到全局和。

![](C:\Users\86184\AppData\Roaming\marktext\images\2025-08-10-22-50-40-image.png)

4. ###### 写入结果
   
   1. 经过第3步中的BlockReduce操作，我们已经得到了整块数据输入的平方和。接下来，将利用该平方和计算RMSNorm中的缩放系数（scale）。
   
   2. 随后，使用该缩放系数对当前线程所负责的所有输入元素进行归一化处理。在此过程中，还需将归一化后的结果逐元素乘以对应的权重（weight），从而得到最终的输出。

```C++
const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float x = block_in[i] * wei[i];
    block_out[i] = x * scale;}
```

原本每个线程每次加载一个标量数据，现在通过使用向量化加载指令（如 `float2`、`float4` 类型读取），使单个线程在一个内存事务中读取多个连续的数据元素。

1. ###### 对输入部分的改写

我们知道，一个批次的输入数据维度为 `size`。因此，我们仍如前所述，定位到当前批次起始的 `block_in` 和 `block_out`。此外，考虑到每次读取的数据大小为 4（即向量化宽度），我们将数据以 4 个为一组进行打包，由此可计算出所需的打包次数 `pack_num`。

<mark>由于输入维度 `size` 不一定是 4 的倍数，我们通过 `pack_size × pack_num` 计算出 `pack_off`，即不超过 `size` 的最大 4 的倍数。剩余部分从 `pack_off` 到 `size`，则逐个加载处理。</mark>

```C++
__global__ void row_rmsnorm_f32_dim_simd(float* in, float* wei, float* out,
                                         int batch, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= batch) {
    return;  }
  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;
```

2. ###### 计算部分

在计算时，我们将输入打包为 `float4`，每次可一次性读取四个元素，即加载为 `in_float4`。

```C++
float sum = 0.0f;
float4* in_pack = reinterpret_cast<float4*>(block_in);
for (int i = tid; i < pack_num; i += blockDim.x) {
   float4 in_float4 = *(in_pack + i);
   sum += in_float4.x * in_float4.x;
   sum += in_float4.y * in_float4.y;
   sum += in_float4.z * in_float4.z;
   sum += in_float4.w * in_float4.w;}
```

3. ###### 对输出部分的改写

在输出部分，我们仍以 4 个 float 为一组进行加载，对每个元素分别乘以对应的缩放因子 `scale` 和位置权重。计算完成后，将结果重新打包为 `float4`，使用 `make_float4` 构造，并写回输出位置。

```C++
for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) = make_float4(
        scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y;
        scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
}
```
