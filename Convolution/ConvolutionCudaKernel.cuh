#ifndef _CONVOLUTIONCUDAKERNEL_CU_
#define _CONVOLUTIONCUDAKERNEL_CU_

#define BLOCK_DIM 32

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void d_transpose(float *idata, float *odata, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = (yIndex * width + xIndex);
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

    if ((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = (yIndex * height + xIndex) * 4;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

__global__ void d_boxfilter(float *id, float *od, int w, int h, int r)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x >= w) return;
    
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[x * 4] * r;

    for (int y = 0; y < (r + 1); y++)
    {
        t += id[(y * w + x) * 4];
    }

    od[x] = t * scale;

    for (int y = 1; y < (r + 1); y++)
    {
        t += id[((y + r) * w + x) * 4];
        t -= id[x * 4];
        od[y * w + x] = t * scale;
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++)
    {
        t += id[((y + r) * w + x) * 4];
        t -= id[(((y - r) * w + x) - w) * 4];
        od[y * w + x] = t * scale;
    }

    // do right edge
    for (int y = h - r; y < h; y++)
    {
        t += id[((h-1) * w + x) * 4];
        t -= id[(((y - r) * w + x) - w) * 4];
        od[y * w + x] = t * scale;
        
    }
}

__global__ void d_simpleRecursive(float *id, float *od, int w, int h, float blur)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    
    const float nsigma = blur < 0.1f ? 0.1f : blur,
	alpha = 1.695f / nsigma,
	ema = (float)exp(-alpha);

    if (x >= w) return;

    id += x * 4;    // advance pointers to correct column
    od += x;

    // forward pass
    float yp = *id;  // previous output

    for (int y = 0; y < h; y++)
    {
        float xc = *id;
        float yc = xc + ema*(yp - xc);
        *od = yc;
        id += w * 4;
        od += w;    // move to next row
        yp = yc;
    }

    // reset pointers to point to last element in column
    id -= w * 4;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    yp = *id;

    for (int y = h-1; y >= 0; y--)
    {
        float xc = *id;
        float yc = xc + ema*(yp - xc);
        *od = (*od + yc) * 0.5f;
        id -= w * 4;
        od -= w;  // move to previous row
        yp = yc;
    }
}

__global__ void d_recursiveGaussian(float *id, float *od, int w, int h, float blur)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    
    const float nsigma = blur < 0.1f ? 0.1f : blur,
	alpha = 1.695f / nsigma,
	ema = (float)exp(-alpha);
	float ema2 = (float)exp(-2*alpha),
	b1 = -2*ema,
	b2 = ema2;
	float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;
	const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
	a0 = k;
	a1 = k*(alpha-1)*ema;
	a2 = k*(alpha+1)*ema;
	a3 = -k*ema2;
	coefp = (a0+a1)/(1+b1+b2);
	coefn = (a2+a3)/(1+b1+b2);

    if (x >= w) return;

    id += x * 4;    // advance pointers to correct column
    od += x;

    // forward pass
    float xp, yp, yb;
    xp = *id;
    yb = coefp*xp;
    yp = yb;

    for (int y = 0; y < h; y++)
    {
        float xc = *id;
        float yc = a0*xc + a1*xp - b1*yp - b2*yb;
        *od = yc;
        id += w * 4;
        od += w;    // move to next row
        xp = xc;
        yb = yp;
        yp = yc;
    }

    // reset pointers to point to last element in column
    id -= w * 4;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    float xn, xa, yn, ya;
    
    xn = xa = *id;
    yn = coefn*xn;
    ya = yn;

    for (int y = h-1; y >= 0; y--)
    {
        float xc = *id;
        float yc = a2*xn + a3*xa - b1*yn - b2*ya;
        xa = xn;
        xn = xc;
        ya = yn;
        yn = yc;
        *od = *od + yc;
        id -= w * 4;
        od -= w;  // move to previous row
    }
}

__global__ void FrequencySharpen(int p_Width, int p_Height, float* p_Input, float* p_Output, float sharpen, int p_Display)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   
   float offset = p_Display == 1 ? 0.5f : 0.0f;
    

   if ((x < p_Width) && (y < p_Height))
   {
      const int index = ((y * p_Width) + x) * 4;
       																												
	 p_Input[index] = (p_Input[index] - p_Output[index]) * sharpen + offset;
	 
	 
	 if (p_Display == 1) {
	 p_Output[index] = p_Input[index];
	 }
   } 
	 
}

__global__ void FrequencyAdd(int p_Width, int p_Height, float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   
    
   if ((x < p_Width) && (y < p_Height))
   {
      const int index = ((y * p_Width) + x) * 4;
       																												
	 p_Output[index] = p_Input[index] + p_Output[index];
   }
}

__global__ void EdgeDetectAdd(int p_Width, int p_Height, float* p_Input, float* p_Output, float p_Threshold)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   
   p_Threshold += 3.0f;
   p_Threshold *= 3.0f;
    
   if ((x < p_Width) && (y < p_Height))
   {
      const int index = ((y * p_Width) + x) * 4;
       																												
	 p_Output[index] = (p_Input[index] - p_Output[index]) * p_Threshold;
   }
}

__global__ void EdgeEnhance(int p_Width, int p_Height, float* p_Input, float* p_Output, float p_Enhance)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
	 const int index = ((y * p_Width) + x) * 4;
	 int X = x == 0 ? 1 : x;
	 int indexE = ((y * p_Width) + X) * 4;
       																												
	 p_Output[index + 0] = (p_Input[index + 0] - p_Input[indexE - 4]) * p_Enhance;
	 p_Output[index + 1] = (p_Input[index + 1] - p_Input[indexE - 3]) * p_Enhance;
	 p_Output[index + 2] = (p_Input[index + 2] - p_Input[indexE - 2]) * p_Enhance;
   }
}

__global__ void ErosionSharedStep1(float * src, float * dst, int radio, int width, int height, int tile_w, int tile_h) {
    extern __shared__ float smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx - radio;
    int y = by * tile_h + ty;
    smem[ty * blockDim.x + tx] = 1.0f;
    __syncthreads();
    if (x < 0 || x >= width || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[(y * width + x) * 4];
    __syncthreads();
    if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w)) {
        return;
    }
    float* smem_thread = &smem[ty * blockDim.x + tx - radio];
    float val = smem_thread[0];
    for (int xx = 1; xx <= 2 * radio; xx++) {
        val = fmin(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}

__global__ void ErosionSharedStep2(float * src, float * dst, int radio, int width, int height, int tile_w, int tile_h) {
    extern __shared__ float smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx;
    int y = by * tile_h + ty - radio;
    smem[ty * blockDim.x + tx] = 1.0f;
    __syncthreads();
    if (x >= width || y < 0 || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (y < (by * tile_h) || y >= ((by + 1) * tile_h)) {
        return;
    }
    float * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
    float val = smem_thread[0];
    for (int yy = 1; yy <= 2 * radio; yy++) {
        val = fmin(val, smem_thread[yy * blockDim.x]);
    }
    dst[(y * width + x) * 4] = val;
}

__global__ void DilateSharedStep1(float * src, float * dst, int radio, int width, int height, int tile_w, int tile_h) {
    extern __shared__ float smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx - radio;
    int y = by * tile_h + ty;
    smem[ty * blockDim.x + tx] = 0.0f;
    __syncthreads();
    if (x < 0 || x >= width || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[(y * width + x) * 4];
    __syncthreads();
    if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w)) {
        return;
    }
    float* smem_thread = &smem[ty * blockDim.x + tx - radio];
    float val = smem_thread[0];
    for (int xx = 1; xx <= 2 * radio; xx++) {
        val = fmax(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}

__global__ void DilateSharedStep2(float * src, float * dst, int radio, int width, int height, int tile_w, int tile_h) {
    extern __shared__ float smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx;
    int y = by * tile_h + ty - radio;
    smem[ty * blockDim.x + tx] = 0.0f;
    __syncthreads();
    if (x >= width || y < 0 || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (y < (by * tile_h) || y >= ((by + 1) * tile_h)) {
        return;
    }
    float * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
    float val = smem_thread[0];
    for (int yy = 1; yy <= 2 * radio; yy++) {
        val = fmax(val, smem_thread[yy * blockDim.x]);
    }
    dst[(y * width + x) * 4] = val;
}

__global__ void CustomMatrix(int p_Width, int p_Height, float* p_Input, float* p_Output, float p_Scale, int p_Normalise, float p_Matrix11, float p_Matrix12, 
			float p_Matrix13, float p_Matrix21, float p_Matrix22, float p_Matrix23, float p_Matrix31, float p_Matrix32, float p_Matrix33)
{  
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   
   p_Scale += 1.0f;
   float normalise = 1.0f;
   if (p_Scale > 1.0f) {
   p_Matrix11 *= p_Scale;
   p_Matrix12 *= p_Scale;
   p_Matrix13 *= p_Scale;
   p_Matrix21 *= p_Scale;
   p_Matrix22 *= p_Scale;
   p_Matrix23 *= p_Scale;
   p_Matrix31 *= p_Scale;
   p_Matrix32 *= p_Scale;
   p_Matrix33 *= p_Scale;
   }
   float total = p_Matrix11 + p_Matrix12 + p_Matrix13 + p_Matrix21 + p_Matrix22 + p_Matrix23 + p_Matrix31 + p_Matrix32 + p_Matrix33;
   if (p_Normalise == 1 && total > 1.0f) {normalise /= total;} 
   
   if ((x < p_Width) && (y < p_Height))
   {
	 const int index = ((y * p_Width) + x) * 4;
	 
	 int start_y = max(y - 1, 0);
     int end_y = min(p_Height - 1, y + 1);
     int start_x = max(x - 1, 0);
     int end_x = min(p_Width - 1, x + 1);
	 
	 p_Output[index] = (p_Input[(end_y * p_Width + start_x) * 4] * p_Matrix11 +
					   p_Input[(end_y * p_Width + x) * 4] * p_Matrix12 +
					   p_Input[(end_y * p_Width + end_x) * 4] * p_Matrix13 +
					   p_Input[(y * p_Width + start_x) * 4] * p_Matrix21 +
					   p_Input[(y * p_Width + x) * 4] * p_Matrix22 +
					   p_Input[(y * p_Width + end_x) * 4] * p_Matrix23 +
					   p_Input[(start_y * p_Width + start_x) * 4] * p_Matrix31 +
					   p_Input[(start_y * p_Width + x) * 4] * p_Matrix32 +
					   p_Input[(start_y * p_Width + end_x) * 4] * p_Matrix33) * normalise;
	}
}

__global__ void Scatter(float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Range, float p_Mix)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
   const int index = (y * p_Width + x) * 4;
   
   float rg = p_Range + 1;
  
   int totA = round((p_Input[index + 0] + p_Input[index + 1] + p_Input[index + 2]) * 1111) + x;
   int totB = round((p_Input[index + 0] + p_Input[index + 1]) * 1111) + y;
  
   int polarityA = fmodf(totA, 2) > 0.0f ? -1.0f : 1.0f;
   int polarityB = fmodf(totB, 2) > 0.0f ? -1.0f : 1.0f;
   int scatterA = fmodf(totA, rg) * polarityA;
   int scatterB = fmodf(totB, rg) * polarityB;

   int X = (x + scatterA) < 0 ? abs(x + scatterA) : ((x + scatterA) > (p_Width - 1) ? (2 * (p_Width - 1)) - (x + scatterA) : (x + scatterA));
   int Y = (y + scatterB) < 0 ? abs(y + scatterB) : ((y + scatterB) > (p_Height - 1) ? (2 * (p_Height - 1)) - (y + scatterB) : (y + scatterB));
																												 
   p_Output[index + 0] = p_Input[((Y * p_Width) + X) * 4 + 0] * (1.0f - p_Mix) + p_Mix * p_Input[index + 0];
   p_Output[index + 1] = p_Input[((Y * p_Width) + X) * 4 + 1] * (1.0f - p_Mix) + p_Mix * p_Input[index + 1];
   p_Output[index + 2] = p_Input[((Y * p_Width) + X) * 4 + 2] * (1.0f - p_Mix) + p_Mix * p_Input[index + 2];
   p_Output[index + 3] = p_Input[index + 3];
   }
}

__global__ void SimpleKernel(int p_Width, int p_Height, float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
	const int index = ((y * p_Width) + x) * 4;
       																												
	p_Output[index] = p_Input[index];
   }
}

#endif // #ifndef _GAUSSIAN_KERNEL_H_