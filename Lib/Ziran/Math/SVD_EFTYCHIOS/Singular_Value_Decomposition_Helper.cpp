//#####################################################################
// Copyright (c) 2009-2011, Eftychios Sifakis.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//   * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
//     other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//#####################################################################

#include "Singular_Value_Decomposition_Helper.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include "PTHREAD_QUEUE.h"

// #define USE_SCALAR_IMPLEMENTATION
// #define USE_SSE_IMPLEMENTATION
#define USE_AVX_IMPLEMENTATION
// #define PRINT_DEBUGGING_OUTPUT

#define COMPUTE_V_AS_MATRIX
// #define COMPUTE_V_AS_QUATERNION
#define COMPUTE_U_AS_MATRIX
// #define COMPUTE_U_AS_QUATERNION

#include "Singular_Value_Decomposition_Preamble.hpp"

using namespace Singular_Value_Decomposition;
using namespace PhysBAM;

//#####################################################################
// Function Run_Index_Range
//#####################################################################
template <class T>
void Singular_Value_Decomposition_Size_Specific_Helper<T>::
    Run_Index_Range(const int imin, const int imax_plus_one)
{

#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"

#ifdef USE_SSE_IMPLEMENTATION
#define STEP_SIZE 4
#endif
#ifdef USE_AVX_IMPLEMENTATION
#define STEP_SIZE 4
#endif
#ifdef USE_SCALAR_IMPLEMENTATION
#define STEP_SIZE 1
#endif

    for (int index = imin; index < imax_plus_one; index += STEP_SIZE) {

        ENABLE_AVX_IMPLEMENTATION(Va11 = _mm256_loadu_pd(a11 + index);)
        ENABLE_AVX_IMPLEMENTATION(Va21 = _mm256_loadu_pd(a21 + index);)
        ENABLE_AVX_IMPLEMENTATION(Va31 = _mm256_loadu_pd(a31 + index);)
        ENABLE_AVX_IMPLEMENTATION(Va12 = _mm256_loadu_pd(a12 + index);)
        ENABLE_AVX_IMPLEMENTATION(Va22 = _mm256_loadu_pd(a22 + index);)
        ENABLE_AVX_IMPLEMENTATION(Va32 = _mm256_loadu_pd(a32 + index);)
        ENABLE_AVX_IMPLEMENTATION(Va13 = _mm256_loadu_pd(a13 + index);)
        ENABLE_AVX_IMPLEMENTATION(Va23 = _mm256_loadu_pd(a23 + index);)
        ENABLE_AVX_IMPLEMENTATION(Va33 = _mm256_loadu_pd(a33 + index);)

#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"

        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(u11 + index, Vu11);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(u21 + index, Vu21);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(u31 + index, Vu31);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(u12 + index, Vu12);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(u22 + index, Vu22);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(u32 + index, Vu32);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(u13 + index, Vu13);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(u23 + index, Vu23);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(u33 + index, Vu33);)

        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(v11 + index, Vv11);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(v21 + index, Vv21);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(v31 + index, Vv31);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(v12 + index, Vv12);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(v22 + index, Vv22);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(v32 + index, Vv32);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(v13 + index, Vv13);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(v23 + index, Vv23);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(v33 + index, Vv33);)

        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(sigma1 + index, Va11);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(sigma2 + index, Va22);)
        ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_pd(sigma3 + index, Va33);)
    }

#undef STEP_SIZE
}
//#####################################################################
namespace Singular_Value_Decomposition {
template class Singular_Value_Decomposition_Size_Specific_Helper<double>;
}
