CUDA implementation of RGB to grayscale.
Roughly 5x to 30x faster than OpenCV's implementation

All functionality is contained in CUDARGB2Y.h and CUDARGB2Y.cu

make -f Makefile

## CUDA based implementation for RGB to Grayscale conversion.
### Implemented a parallel algorithm from conversion of an image in RGB colour scheme to grayscale colour scheme.

RGB to Grayscale conversion is an elementary task in computer vision. In such a conversion an image is represented as a matrix of pixel values which is then manipulated for the said operation. The aforementioned manipulation can be decomposed into smaller individual tasks which can be independently operated upon. Our implementation performs the said operation about 6-30 times faster than the prevalent library OpenCV used for such conversions. Our algorithm divides the entire image matrix into smaller sections each of which is fed into a GPU grid and every pixel value is then handled by a single thread block.

### H/W & S/W Requirements

CUDA Toolkit 10.0 (Nvidia CUDA Compiler), Nvidia GPU Driver CUDA version 440, Compatible Nvidia GPU, OpenCV.

### Read the following for conceptual clarity

RGB is a colour scheme representation where any image is represented as 3-dimensional matrix (3 two-dimensional arrays) where each dimension corresponds to the colours Red, Blue & Green respectively. The elements in the matrix correspond to the intensity of the corresponding colour in the given pixel that is the element [i,j] for all 3 channels is the intensity value of that colour in the  [i,j] pixel in the image. Thus, RGB represents the image as a combination of the three aforementioned colours with the element values corresponding to the intensity of that colour at that pixel in the image.
Grayscale is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray. The contrast ranges from black at the weakest intensity to white at the strongest.
                         
#### Methods for conversion of RGB to Grayscale:

Equal conversion: Average method is the most simple one. We just have to take the average of three colors. Since it's an RGB image, it means that we have to add R with G with B and then divide it by 3 to get the desired grayscale image.
Grayscale=(R+G+B)/3
Weighted Conversion: In the aforementioned method, all three colour channels are assigned the same weightage. However, the resulting image is not proper as shown below. This is due to the fact that the three colours have different wavelengths and therefore that factor is completely ignored in the previous step. To account for that we introduce a weighted averaging method where each colour is assigned weights commensurate to their wavelengths.
Grayscale = ( (0.3 * R) + (0.59 * G) + (0.11 * B) )
Weights can be varied as per implementation.

#### Shared Memory
Striding through global memory is problematic regardless of the generation of the CUDA hardware, and would seem to be unavoidable in many cases, such as when accessing elements in a multidimensional array along the second and higher dimensions. However, it is possible to coalesce memory access in such cases if we use shared memory.
Because it is on-chip, shared memory is much faster than local and global memory. In fact, shared memory latency is roughly 100x lower than uncached global memory latency (provided that there are no bank conflicts between the threads, which we will examine later in this post). Shared memory is allocated per thread block, so all threads in the block have access to the same shared memory. Threads can access data in shared memory loaded from global memory by other threads within the same thread block. This capability (combined with thread synchronization) has a number of uses, such as user-managed data caches, high-performance cooperative parallel algorithms (parallel reductions, for example), and to facilitate global memory coalescing in cases where it would otherwise not be possible
When sharing data between threads, we need to be careful to avoid race conditions, because while threads in a block run logically in parallel, not all threads can execute physically at the same time. To ensure correct results when parallel threads cooperate, we must synchronize the threads. CUDA provides a simple barrier synchronization primitive, __syncthreads(). 
A thread’s execution can only proceed past a __syncthreads() after all threads in its block have executed the __syncthreads(). Thus, we can avoid the race condition described above by calling __syncthreads() after the store to shared memory and before any threads load from shared memory. It’s important to be aware that calling __syncthreads() in divergent code is undefined and can lead to deadlock—all threads within a thread block must call __syncthreads() at the same point.
For purposes of configuring the amount of shared memory,on devices of compute capability 2.x and 3.x, each multiprocessor has 64KB of on-chip memory that can be partitioned between L1 cache and shared memory. For devices of compute capability 2.x, there are two settings, 48KB shared memory / 16KB L1 cache, and 16KB shared memory / 48KB L1 cache. By default the 48KB shared memory setting is used. This can be configured during runtime API from the host (CPU) for all kernels using cudaDeviceSetCacheConfig() or on a per-kernel basis using cudaFuncSetCacheConfig(). These accept one of three options: 
cudaFuncCachePreferNone, cudaFuncCachePreferShared & cudaFuncCachePreferL1. 
The driver (GPU) will honor the specified preference except when a kernel requires more shared memory per thread block than available in the specified configuration. 

#### Texture Memory
Texture memory is a cached on-chip sophisticated read-only, which provides higher effective bandwidth by reducing memory requests to off-chip DRAM. Specifically, texture caches are designed for graphics applications where memory access patterns exhibit a great deal of spatial locality. In a computing application, this roughly implies that a thread is likely to read from an address “near” the address that nearby threads read, as shown in figure,

Arithmetically, the four addresses shown are not consecutive, so they would not be cached together in a typical CPU caching scheme. But since GPU texture caches are designed to accelerate access patterns such as this one, you will see an increase in performance in this case when using texture memory instead of global memory.
The read-only texture memory space is cached. Therefore, a texture fetch costs one device memory read only on a cache miss; otherwise, it just costs one read from the texture cache. The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture addresses that are close together will achieve best performance. Texture memory is also designed for streaming fetches with a constant latency; that is, a cache hit reduces DRAM bandwidth demand, but not fetch latency.
The process of reading a texture is called a texture fetch and is done using Texture Functions. The first parameter of a texture fetch specifies an object called a texture reference. A texture reference defines which part of texture memory is fetched. A texture reference has several attributes. One of them is its dimensionality that specifies whether the texture is addressed as a one-dimensional array using one texture coordinate, a two-dimensional array using two texture coordinates, or a three-dimensional array using three texture coordinates. Elements of the array are called texels, short for “texture elements.” The type of a Texel is restricted to the basic integer and single-precision floating-point types and any of the 1-, 2-, and 4-component vector types Textures are fetched using tex1D(), tex2D(), or tex3D() rather than tex1Dfetch(), so that the hardware provides other capabilities that are useful for some applications such as image processing


Texture memory can provide additional speedups if we utilize some of the conversions that texture samplers can perform automatically, such as unpacking packed data into separate variables or converting 8- and 16-bit integers to normalized floating-point numbers.

#### Implementation
In the project, following have been used
* uint8_t datatype for declaring identifiers. It is used for integers of exactly 8 bits width. Suitable for our project as RGB values vary from 0-255.
* Shared Memory as allows for faster memory access speeds thereby allowing faster execution of program
* Texture Memory as our image is represented as a 2-dimensional array and for the same, texture memory allows faster access times & lower latency
* Left & Right Shift Operators for manipulating index values to access pixel values. It was preferred over division operator as the division operation is   computationally expensive & time consuming as compared to shift operators.

Input: Image in RGB Colour Format
Output: Image in Grayscale

