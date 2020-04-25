//#####################################################################
// Copyright (c) 2010-2011, Eftychios Sifakis.
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

//###########################################################
// Compute the Givens angle (and half-angle)
//###########################################################

ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_mul_pd(VS21, Vone_half);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_sub_pd(VS11, VS22);)

ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vsh, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_cmp_pd(Vtmp2, Vtiny_number, _CMP_GE_OS);)
ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_and_pd(Vtmp1, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vch = _mm256_blendv_pd(Vone, Vtmp5, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vsh, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vch, Vch);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_add_pd(Vtmp1, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_sqrt_pd(Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_div_pd(Vone, Vtmp4);)

ENABLE_AVX_IMPLEMENTATION(Vs = _mm256_mul_pd(Vtmp4, Vone_half);)
ENABLE_AVX_IMPLEMENTATION(Vc = _mm256_mul_pd(Vtmp4, Vs);)
ENABLE_AVX_IMPLEMENTATION(Vc = _mm256_mul_pd(Vtmp4, Vc);)
ENABLE_AVX_IMPLEMENTATION(Vc = _mm256_mul_pd(Vtmp3, Vc);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_add_pd(Vtmp4, Vs);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_sub_pd(Vtmp4, Vc);)

ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_mul_pd(Vtmp4, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vch = _mm256_mul_pd(Vtmp4, Vch);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vfour_gamma_squared, Vtmp1);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_cmp_pd(Vtmp2, Vtmp1, _CMP_LE_OS);)

ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_blendv_pd(Vsh, Vsine_pi_over_eight, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vch = _mm256_blendv_pd(Vch, Vcosine_pi_over_eight, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vsh, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vch, Vch);)
ENABLE_AVX_IMPLEMENTATION(Vc = _mm256_sub_pd(Vtmp2, Vtmp1);)
ENABLE_AVX_IMPLEMENTATION(Vs = _mm256_mul_pd(Vch, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vs = _mm256_add_pd(Vs, Vs);)

//###########################################################
// Perform the actual Givens conjugation
//###########################################################

#ifndef USE_ACCURATE_RSQRT_IN_JACOBI_CONJUGATION
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_add_pd(Vtmp1, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(VS33 = _mm256_mul_pd(VS33, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(VS31 = _mm256_mul_pd(VS31, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(VS32 = _mm256_mul_pd(VS32, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(VS33 = _mm256_mul_pd(VS33, Vtmp3);)
#endif

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vs, VS31);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vs, VS32);)
ENABLE_AVX_IMPLEMENTATION(VS31 = _mm256_mul_pd(Vc, VS31);)
ENABLE_AVX_IMPLEMENTATION(VS32 = _mm256_mul_pd(Vc, VS32);)
ENABLE_AVX_IMPLEMENTATION(VS31 = _mm256_add_pd(Vtmp2, VS31);)
ENABLE_AVX_IMPLEMENTATION(VS32 = _mm256_sub_pd(VS32, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vs, Vs);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(VS22, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(VS11, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Vc, Vc);)
ENABLE_AVX_IMPLEMENTATION(VS11 = _mm256_mul_pd(VS11, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(VS22 = _mm256_mul_pd(VS22, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(VS11 = _mm256_add_pd(VS11, Vtmp1);)
ENABLE_AVX_IMPLEMENTATION(VS22 = _mm256_add_pd(VS22, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_sub_pd(Vtmp4, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(VS21, VS21);)
ENABLE_AVX_IMPLEMENTATION(VS21 = _mm256_mul_pd(VS21, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Vc, Vs);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vtmp2, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_mul_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(VS11 = _mm256_add_pd(VS11, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(VS21 = _mm256_sub_pd(VS21, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(VS22 = _mm256_sub_pd(VS22, Vtmp2);)

//###########################################################
// Compute the cumulative rotation, in quaternion form
//###########################################################

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vsh, Vqvvx);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vsh, Vqvvy);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vsh, Vqvvz);)
ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_mul_pd(Vsh, Vqvs);)

ENABLE_AVX_IMPLEMENTATION(Vqvs = _mm256_mul_pd(Vch, Vqvs);)
ENABLE_AVX_IMPLEMENTATION(Vqvvx = _mm256_mul_pd(Vch, Vqvvx);)
ENABLE_AVX_IMPLEMENTATION(Vqvvy = _mm256_mul_pd(Vch, Vqvvy);)
ENABLE_AVX_IMPLEMENTATION(Vqvvz = _mm256_mul_pd(Vch, Vqvvz);)

ENABLE_AVX_IMPLEMENTATION(VQVVZ = _mm256_add_pd(VQVVZ, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vqvs = _mm256_sub_pd(Vqvs, VTMP3);)
ENABLE_AVX_IMPLEMENTATION(VQVVX = _mm256_add_pd(VQVVX, VTMP2);)
ENABLE_AVX_IMPLEMENTATION(VQVVY = _mm256_sub_pd(VQVVY, VTMP1);)
