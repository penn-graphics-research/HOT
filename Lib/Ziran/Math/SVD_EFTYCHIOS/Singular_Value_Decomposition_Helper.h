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

#ifndef __Singular_Value_Decomposition_Helper__
#define __Singular_Value_Decomposition_Helper__

#include <Ziran/CS/Util/Debug.h>
#include <Ziran/CS/Util/Forward.h>
#include <vector>

namespace Singular_Value_Decomposition {

template <class T>
class Singular_Value_Decomposition_Size_Specific_Helper {
    T *const a11, *const a21, *const a31, *const a12, *const a22, *const a32, *const a13, *const a23, *const a33;
    T *const u11, *const u21, *const u31, *const u12, *const u22, *const u32, *const u13, *const u23, *const u33;
    T *const v11, *const v21, *const v31, *const v12, *const v22, *const v32, *const v13, *const v23, *const v33;
    T *const sigma1, *const sigma2, *const sigma3;

public:
    explicit Singular_Value_Decomposition_Size_Specific_Helper(
        T* const a11_input, T* const a21_input, T* const a31_input,
        T* const a12_input, T* const a22_input, T* const a32_input,
        T* const a13_input, T* const a23_input, T* const a33_input,
        T* const u11_input, T* const u21_input, T* const u31_input,
        T* const u12_input, T* const u22_input, T* const u32_input,
        T* const u13_input, T* const u23_input, T* const u33_input,
        T* const v11_input, T* const v21_input, T* const v31_input,
        T* const v12_input, T* const v22_input, T* const v32_input,
        T* const v13_input, T* const v23_input, T* const v33_input,
        T* const sigma1_input, T* const sigma2_input, T* const sigma3_input)
        : a11(a11_input), a21(a21_input), a31(a31_input), a12(a12_input), a22(a22_input), a32(a32_input), a13(a13_input), a23(a23_input), a33(a33_input), u11(u11_input), u21(u21_input), u31(u31_input), u12(u12_input), u22(u22_input), u32(u32_input), u13(u13_input), u23(u23_input), u33(u33_input), v11(v11_input), v21(v21_input), v31(v31_input), v12(v12_input), v22(v22_input), v32(v32_input), v13(v13_input), v23(v23_input), v33(v33_input), sigma1(sigma1_input), sigma2(sigma2_input), sigma3(sigma3_input)
    {
    }

    void Run_Index_Range(const int imin, const int imax_plus_one);
};

template <class T, int dim>
void Singular_Value_Decomposition_Fast(
    std::vector<ZIRAN::Matrix<T, dim, dim>>& Fs,
    std::vector<ZIRAN::Matrix<T, dim, dim>>& Us,
    std::vector<ZIRAN::Vector<T, dim>>& sigmas,
    std::vector<ZIRAN::Matrix<T, dim, dim>>& Vs)
{
    // do SVD
    //ZIRAN_ASSERT(std::is_same<T, double>::value, "SIMD SVD only supports double now!!");
    size_t size = Fs.size();
    size_t padded_size = std::ceil(size / (T)4) * 4;
    if (padded_size > size) Fs.resize(padded_size, ZIRAN::Matrix<T, dim, dim>::Identity());

    T *a11, *a21, *a31, *a12, *a22, *a32, *a13, *a23, *a33;
    T *u11, *u21, *u31, *u12, *u22, *u32, *u13, *u23, *u33;
    T *v11, *v21, *v31, *v12, *v22, *v32, *v13, *v23, *v33;
    T *sigma1, *sigma2, *sigma3;

    void* buffers_raw;
    int buffers_return = 0;
    // alignment 64 : 1 double == 8 bytes == 64 bits
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    a11 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    a21 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    a31 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    a12 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    a22 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    a32 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    a13 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    a23 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    a33 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    u11 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    u21 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    u31 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    u12 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    u22 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    u32 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    u13 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    u23 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    u33 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    v11 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    v21 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    v31 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    v12 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    v22 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    v32 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    v13 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    v23 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    v33 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    sigma1 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    sigma2 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);
    buffers_return = posix_memalign(&buffers_raw, 64, padded_size * 8);
    sigma3 = reinterpret_cast<T*>(buffers_raw);
    ZIRAN_ASSERT(buffers_return == 0);

    // load F from AOS to SOA
    __m256i vBuffer = _mm256_set_epi64x(27, 18, 9, 0);
    for (size_t i = 0; i < padded_size / 4; ++i) {
        __m256d vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(Fs.data()) + i * 36), vBuffer, 8);
        _mm256_store_pd(&a11[i * 4], vFourDouble);

        vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(Fs.data()) + i * 36 + 1), vBuffer, 8);
        _mm256_store_pd(&a21[i * 4], vFourDouble);

        vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(Fs.data()) + i * 36 + 2), vBuffer, 8);
        _mm256_store_pd(&a31[i * 4], vFourDouble);

        vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(Fs.data()) + i * 36 + 3), vBuffer, 8);
        _mm256_store_pd(&a12[i * 4], vFourDouble);

        vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(Fs.data()) + i * 36 + 4), vBuffer, 8);
        _mm256_store_pd(&a22[i * 4], vFourDouble);

        vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(Fs.data()) + i * 36 + 5), vBuffer, 8);
        _mm256_store_pd(&a32[i * 4], vFourDouble);

        vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(Fs.data()) + i * 36 + 6), vBuffer, 8);
        _mm256_store_pd(&a13[i * 4], vFourDouble);

        vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(Fs.data()) + i * 36 + 7), vBuffer, 8);
        _mm256_store_pd(&a23[i * 4], vFourDouble);

        vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(Fs.data()) + i * 36 + 8), vBuffer, 8);
        _mm256_store_pd(&a33[i * 4], vFourDouble);
    }

    using namespace Singular_Value_Decomposition;
    Singular_Value_Decomposition_Size_Specific_Helper<T> test(
        a11, a21, a31, a12, a22, a32, a13, a23, a33,
        u11, u21, u31, u12, u22, u32, u13, u23, u33,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
        sigma1, sigma2, sigma3);

    test.Run_Index_Range(0, padded_size);

    Us.resize(size);
    Vs.resize(size);
    sigmas.resize(size);
    for (int i = 0; i < (int)size; ++i) {
        Us[i](0, 0) = u11[i];
        Us[i](1, 0) = u21[i];
        Us[i](2, 0) = u31[i];
        Us[i](0, 1) = u12[i];
        Us[i](1, 1) = u22[i];
        Us[i](2, 1) = u32[i];
        Us[i](0, 2) = u13[i];
        Us[i](1, 2) = u23[i];
        Us[i](2, 2) = u33[i];

        Vs[i](0, 0) = v11[i];
        Vs[i](1, 0) = v21[i];
        Vs[i](2, 0) = v31[i];
        Vs[i](0, 1) = v12[i];
        Vs[i](1, 1) = v22[i];
        Vs[i](2, 1) = v32[i];
        Vs[i](0, 2) = v13[i];
        Vs[i](1, 2) = v23[i];
        Vs[i](2, 2) = v33[i];

        sigmas[i](0) = sigma1[i];
        sigmas[i](1) = sigma2[i];
        sigmas[i](2) = sigma3[i];
    }

    // release memory
    free(a11);
    free(a21);
    free(a31);
    free(a12);
    free(a22);
    free(a32);
    free(a13);
    free(a23);
    free(a33);
    free(u11);
    free(u21);
    free(u31);
    free(u12);
    free(u22);
    free(u32);
    free(u13);
    free(u23);
    free(u33);
    free(v11);
    free(v21);
    free(v31);
    free(v12);
    free(v22);
    free(v32);
    free(v13);
    free(v23);
    free(v33);
    free(sigma1);
    free(sigma2);
    free(sigma3);
}

} // namespace Singular_Value_Decomposition
#endif
