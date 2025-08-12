Block tile

Thread tile

Float4读取----**减少访存次数、提高内存带宽利用率**

双缓冲机制队列+乒乓缓冲机制----将访存与计算阶段进行重叠---**挖掘显存带宽潜力**

双缓冲队列优化背景

优化前的实现流程可以分为三个阶段：

1. 全局显存 → 共享显存块（As和Bs）

2. 共享显存 → 寄存器（a_frag和b_frag）

3. 计算 → 写回全局显存

我们的优化思路是：**让共享显存到寄存器的数据搬运，以及全局显存到共享显存的数据搬运，与计算过程并行执行**。

### 优化方案

![](C:\Users\86184\AppData\Roaming\marktext\images\2025-08-12-15-05-32-image.png)

###### **共享显存扩容**

将原先单一的As和Bs共享显存划分为两个独立的共享显存块，即As显存块从一块扩展为两块As[2]，Bs显存块同样从一块扩展为两块Bs[2]。<u>通过将共享显存容量扩大一倍，我们可以实现乒乓操作：一个显存块用于当前计算的数据读取，另一个显存块则同时进行数据写入准备。</u>

###### **乒乓缓冲机制**

采用双缓冲策略实现显存块的交替使用：

- 当前轮次：从共享显存块As[0]**读取数据进行计算，同时向共享显存块**As[1]写入下一轮所需数据

- 下一轮次：角色互换，从共享显存块As[1]**读取数据进行计算，向共享显存块**As[0]写入数据

通过这种交替机制，数据搬运与计算得以完全重叠，显著提升了整体执行效率。如上图所示，当计算单元处理As[1]共享显存块中的输入数据时，全局显存可以同步向As[0]块中搬运下一批数据。同理，当As[1]中的输入数据作为操作数A计算完成后，系统会切换使用顺序，开始计算As[0]中的下一块输入数据。

###### 乒乓缓冲机制

在前述双缓冲队列的优化方案中，我们提到了**乒乓缓冲（Ping-Pong Buffering）机制**。该机制的核心思想是利用两块大小相同的共享内存缓冲区，实现**计算与数据预取的重叠**，从而隐藏全局显存访问的延迟。

具体而言，我们为参与矩阵乘法的两个输入矩阵$A$和 $B$ 分别分配两个共享内存缓冲区，通过交替使用这两组缓冲区，在执行当前数据块计算的同时，异步地将下一数据块从全局显存预取到另一个空闲的缓冲区中。

整个优化流程围绕矩阵乘法中沿  $K$ 维度的分块展开，具体步骤如下：

1. **初始化阶段**：在进入  $K$ 轴向的主循环之前，首先将第一个  $BM \times BK$ 的 $ A$ 子块和第一个 $ BN \times BK$的 $ B$ 子块加载到共享内存中，作为初始计算数据。

2. **主循环中的流水线重叠**：在后续的每一轮循环中，我们将**当前分块的计算操作**与**下一相邻分块的数据预取操作**进行重叠：
   
   1. 使用当前缓冲区中的 $ A$ 和 $ B$ 子块执行部分矩阵乘法（通常为 GEMM 内核中的一个分块计算）；
   
   2. 同时，启动对下一个 $ BK$ 深度数据的加载：即从全局显存读取下一个 $ BM \times BK$ 的 $ A$ 块和 $ BN \times BK$ 的 $ B$ 块，写入另一个缓冲区，供下一次迭代使用。

3. ###### 数据预取

我们以步长 $ BK$ 遍历整个 $ K$ 维度。设当前迭代位于 $ k$位置，则本次循环处理的是区间 $[k, k + BK]$ 上的数据块；与此同时，在该次循环开始前或计算过程中，会提前发起对下一个区间 $[k + BK, k + 2BK]$ 数据的加载请求。

```C++
  do {
      k += BK;
      if (k < K) {
        // 读取下一个循环需要计算的A分块数据
        for (int i = 0; i < BM; i += a_tile_stride) {
          int ldg_index = i / a_tile_stride * 4;
          FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
              FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
        }
        // 读取下一个循环需要计算的B分块数据
        for (int i = 0; i < BK; i += b_tile_stride) {
          int ldg_index = i / b_tile_stride * 4;
          FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
              FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
        }
      }
    ... while (k < K);
```

2. ###### 计算本轮数据

当前循环计算的是区间 $[k, k + BK)$ 的 $As$ 和 $Bs$ 数据，而预取的是下一区间 $[k + BK, k + 2BK]$ 的数据，供下一轮使用，实现**计算当前，预取下一段**。

具体计算中，我们将 $TM \times BK$ 的 $As$ 和 $BK \times TN$ 的 $Bs$ 沿 $K$ 维进一步切分为 $TM \times 1$ 与 $1 \times TN$ 的细粒度块，通过外积累加完成矩阵乘。在此过程中，采用**寄存器级流水**：在计算当前切片的同时，预加载下一个切片数据到寄存器，实现计算与寄存器读取的重叠。

综上，系统实现两级流水重叠：

1. **小尺度重叠（寄存器级）**：计算当前 $TM \times1 $ 与 $1 \times T$外积时，预取下一细粒度数据到寄存器。

2. **大尺度重叠（共享内存级）**：使用双缓冲机制，在计算当前 $BK$ 分块的同时，异步加载下一个 $BK$ 块从全局内存到共享内存。

计算当前分块的流程前文已详细说明，此处不再赘述。当前阶段的核心是实现**寄存器级流水线重叠**：在执行当前细粒度计算的同时，预加载下一数据片段到寄存器，以隐藏内存访问延迟。

```C++
for (int bk = 0; bk < BK - 1; bk++) {
  for (int m = 0; m < TM; m += 4) {
    FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) =
      FETCH_FLOAT4(As[load_index][OFFSET(bk + 1, ty + m, BM)]);
  }
  for (int n = 0; n < TN; n += 4) {
    FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) =
      FETCH_FLOAT4(Bs[load_index][OFFSET(bk + 1, tx + n, BN)]);
  }
  for (int m = 0; m < TM; m++) {
    for (int n = 0; n < TN; n++) {
      accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
    }
  }
}
```

3. #### 读写索引调换

我们在第 $k$ 轮循环中计算的是区间 $[k, k + BK)$ 的数据，同时已将下一区间 $[k + BK, k + 2BK)$ 的数据预取到共享内存中。因此，当本轮计算完成时，下一轮所需数据早已就绪，无需额外等待。

此时，我们只需将计算结果（通常暂存在寄存器中）写回对应的输出缓冲区或共享内存，然后交换双缓冲的读写索引——即切换 `write_index` 与 `load_index`，使下一轮循环能正确指向新的输入缓冲区。通过这种方式，在下一次 $K$ 轴循环开始时，系统将自动使用已预加载的新数据块，实现无缝的流水线衔接。

```C++
for (int i = 0; i < BM; i += a_tile_stride) {
  int ldg_index = i / a_tile_stride * 4;
  As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] =
    ldg_a_reg[ldg_index];
  As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] =
    ldg_a_reg[ldg_index + 1];
  As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] =
    ldg_a_reg[ldg_index + 2];
  As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] =
    ldg_a_reg[ldg_index + 3];
}
for (int i = 0; i < BK; i += b_tile_stride) {
  int ldg_index = i / b_tile_stride * 4;
  FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
    FETCH_FLOAT4(ldg_b_reg[ldg_index]);
}
write_index ^= 1;
```

在这我们将之前 $[k+BK, k + 2* BK)$ 上只写到寄存器的数据写到As和Bs中，因为在这里，本轮循环已经接近尾声 $[k, k + BK)$ 范围的数据已经进行计算完毕，我们现在要做的事情就是将当前需要将计算结果从寄存器写回共享内存中的输出缓冲区，同时切换双缓冲的读写角色。

为此，我们通过 **按位异或 1（`^= 1`）** 操作翻转 `write_index`，实现读写缓冲区的乒乓切换：

- 若 `write_index == 0`，则切换为 1，对应缓冲区 1 作为下一写目标；

- 若 `write_index == 1`，则切换为 0，写入目标切回缓冲区 0。

与此同时，`load_index` 自动指向另一个缓冲区（即原 `write_index` 的旧值），确保在下一轮$K$轴迭代开始时，能直接读取**已预加载完成**的下一批 $As$和 $Bs$ 数据。

![](C:\Users\86184\AppData\Roaming\marktext\images\2025-08-12-23-42-40-image.png)
