## 关于向量乘法的优化思考

向量乘是点对点的操作，相关的优化方法比矩阵乘法简单很多。

我可以想到两种角度来优化：**提高运算时的访存速度**和**提高数据传输的I/O速度**。

### （一）提高运算时的访存速度：共享内存

使用共享内存可以显著提高运算时的访存速度CUDA程序的性能，因为共享内存的访问速度比全局内存要快得多。

共享内存是芯片内存，位于每个Streaming Multiprocessor (SM) 内部，访问速度比芯片外的全局内存要快得多。因此，在计算过程中利用共享内存进行数据缓存和中间结果存储，可以大幅度提升性能。

__1. 首先声明共享内存:__

```cuda
extern __shared__ float shared_data[];
float* shared_A = shared_data; // 指向共享内存的前半部分
float* shared_B = shared_data + blockDim.x; // 指向共享内存的后半部分
```

__2. 将数据从全局内存拷贝到共享内存:__

```cuda
if (idx < N) {
    shared_A[threadIdx.x] = A[idx];
    shared_B[threadIdx.x] = B[idx];
}
__syncthreads(); // 同步，确保所有线程都拷贝完数据
```

__3. 在所有线程都完成数据拷贝后，线程从共享内存中读取数据进行计算。__

### （二）提高数据传输的I/O速度

有两种方法：**使用页锁定内存**和**异步内存传输和流**

CUDA提供了`cudaMallocHost`函数来分配页锁定（pinned）内存，这种内存不能被操作系统分页到磁盘，因此可以显著提高主机和设备之间的内存传输速度：

```cuda
float* h_A;
cudaMallocHost(&h_A, size);
```

使用异步内存传输（`cudaMemcpyAsync`）可以让数据传输和计算重叠进行，从而提高并行性。需要结合流（streams）来使用异步传输:

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
// 执行其他操作，或者在另一个流中进行计算
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```
