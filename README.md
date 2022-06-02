# optimization-playGround
Applying code optimization based on hardware resources,
Making a simple image processing library to test different approaches


Optimized mini‐img‐processing libary implemented from scratch using C, Pthreads, And Intel‐intrinsics.


## Results 
![](/imgModule/img_module/benchMarking2.png)

- Sequential thresholding : linear pass 
- unrolled thresholding : using loop unrolling  
- Fast thresholding : using loop unrolling + precomputed addresses 
- SSE2 thresholding : using vectorization 
- parallel thresholding : using threads + SSE2 

- First excutable "imgModuleRelease" is compiled with '-msse2' '-O3' flags   