#ifndef _QUALIFIERCUDAKERNEL_CU_
#define _QUALIFIERCUDAKERNEL_CU_

#define BLOCK_DIM 32

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ float Luma(float R, float G, float B, int L) {
  float lumaRec709 = R * 0.2126f + G * 0.7152f + B * 0.0722f;
  float lumaRec2020 = R * 0.2627f + G * 0.6780f + B * 0.0593f;
  float lumaDCIP3 = R * 0.209492f + G * 0.721595f + B * 0.0689131f;
  float lumaACESAP0 = R * 0.3439664498f + G * 0.7281660966f + B * -0.0721325464f;
  float lumaACESAP1 = R * 0.2722287168f + G * 0.6740817658f + B * 0.0536895174f;
  float lumaAvg = (R + G + B) / 3.0f;
  float lumaMax = fmax(fmax(R, G), B);
  float Lu = L == 0 ? lumaRec709 : L == 1 ? lumaRec2020 : L == 2 ? lumaDCIP3 : L == 3 ? lumaACESAP0 : L == 4 ? lumaACESAP1 : L == 5 ? lumaAvg : lumaMax;
  return Lu;
  }
  
__device__ float Alpha(float p_ScaleA, float p_ScaleB, float p_ScaleC, float p_ScaleD, float p_ScaleE, float p_ScaleF, float N, int p_Switch) {
  float r = p_ScaleA;				
  float g = p_ScaleB;				
  float b = p_ScaleC;				
  float a = p_ScaleD;				
  float d = 1.0f / p_ScaleE;							
  float e = 1.0f / p_ScaleF;											 
  float w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= N ? 1.0f : (r >= N ? powf((r - N) / (1.0f - g), d) : 0.0f));		
  float k = a == 1.0f ? 0.0f : (a + b <= N ? 1.0f : (a <= N ? powf((N - a) / b, e) : 0.0f));						
  float alpha = k * w;											 
  float alphaV = p_Switch==1 ? 1.0f - alpha : alpha;
  return alphaV;
  }
      
__device__ void RGB_to_HSV( float r, float g, float b, float *h, float *s, float *v ) {
  float min = fmin(fmin(r, g), b);
  float max = fmax(fmax(r, g), b);
  *v = max;
  float delta = max - min;
  if (max != 0.) {
	  *s = delta / max;
  } else {
	  *s = 0.f;
	  *h = 0.f;
	  return;
  }
  if (delta == 0.) {
	  *h = 0.f;
  } else if (r == max) {
	  *h = (g - b) / delta;
  } else if (g == max) {
	  *h = 2 + (b - r) / delta;
  } else {
	  *h = 4 + (r - g) / delta;
  }
  *h *= 1.0f / 6.;
  if (*h < 0) {
	  *h += 1.0f;
  }
  }

__device__ void HSV_to_RGB(float H, float S, float V, float *r, float *g, float *b)
  {
  if (S == 0.0f) {
	  *r = *g = *b = V;
	  return;
  }
  H *= 6.0f;
  int i = floor(H);
  float f = H - i;
  i = (i >= 0) ? (i % 6) : (i % 6) + 6;
  float p = V * (1.0f - S);
  float q = V * (1.0f - S * f);
  float t = V * (1.0f - S * (1.0f - f));
  *r = i == 0 ? V : i == 1 ? q : i == 2 ? p : i == 3 ? p : i == 4 ? t : V;
  *g = i == 0 ? t : i == 1 ? V : i == 2 ? V : i == 3 ? q : i == 4 ? p : p;
  *b = i == 0 ? p : i == 1 ? p : i == 2 ? t : i == 3 ? V : i == 4 ? V : q;
  }
  
__global__ void d_garbageCore(float* p_Input, int p_Width, int p_Height, float p_Garbage, float p_Core)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < p_Width) && (y < p_Height))
	{

	const int index = ((y * p_Width) + x) * 4;
	
	float A = p_Input[index];
	
	if (p_Core > 0.0f) {
	float CoreA = fmin(A * (1.0f + p_Core), 1.0f);
	CoreA = fmax(CoreA + (0.0f - p_Core * 3.0f) * (1.0f - CoreA), 0.0f);
	A = fmax(A, CoreA);
	}
	
	if (p_Garbage > 0.0f) {
	float GarA = fmax(A + (0.0f - p_Garbage * 3.0f) * (1.0f - A), 0.0f);
	GarA = fmin(GarA * (1.0f + p_Garbage* 3.0f), 1.0f);
	A = fmin(A, GarA);
	}
	
	p_Input[index] = A;
	
	}
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

__global__ void d_transposeDiagonal(float *idata, float *odata, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

    int blockIdx_x, blockIdx_y;

    // do diagonal reordering
    if (width == height)
    {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    }
    else
    {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
    // and similarly for y
    
    // read the matrix tile into shared memory
    int xIndex = blockIdx_x * BLOCK_DIM + threadIdx.x;
    int yIndex = blockIdx_y * BLOCK_DIM + threadIdx.y;
    
    if ((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = (yIndex * width + xIndex);
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx_y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx_x * BLOCK_DIM + threadIdx.y;

    if ((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = (yIndex * height + xIndex) * 4;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
    
}

__global__ void d_recursiveGaussian(float *id, float *od, int w, int h, float a0, float a1, float a2, float a3, float b1, float b2, float coefp, float coefn)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

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

__global__ void SimpleKernel(int p_Width, int p_Height, float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
      const int index = ((y * p_Width) + x) * 4;
       																												
	 p_Output[index + 0] = p_Input[index + 0];
	 p_Output[index + 1] = p_Input[index + 1];
	 p_Output[index + 2] = p_Input[index + 2];
	 p_Output[index + 3] = p_Input[index + 3];
   }
}

#endif // #ifndef _GAUSSIAN_KERNEL_H_