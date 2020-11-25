#pragma once

#include <cstdint>

#include "cuda_runtime.h"

#ifdef __INTELLISENSE__
#define asm(x)
#define min(x) 0
#define fmaf(x) 0
#include "device_launch_parameters.h"
#define __CUDACC__
#include "device_functions.h"
#undef __CUDACC__
#endif

void CUDARGB2Y(bool weight, const cudaTextureObject_t tex_img, const int pixels, uint8_t* const __restrict d_newimg);
