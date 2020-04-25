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
// Compute the Givens half-angle, construct the Givens quaternion and the rotation sine/cosine (for the full angle)
//###########################################################

ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_mul_pd(VANPIVOT, VANPIVOT);)
ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_cmp_pd(Vsh, Vsmall_number, _CMP_GE_OS);)
ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_and_pd(Vsh, VANPIVOT);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vtmp5, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vch = _mm256_sub_pd(Vtmp5, VAPIVOT);)
ENABLE_AVX_IMPLEMENTATION(Vch = _mm256_max_pd(Vch, VAPIVOT);)
ENABLE_AVX_IMPLEMENTATION(Vch = _mm256_max_pd(Vch, Vsmall_number);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_cmp_pd(VAPIVOT, Vtmp5, _CMP_GE_OS);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vch, Vch);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vsh, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp1, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_sqrt_pd(Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_div_pd(Vone, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Vtmp1, Vone_half);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp1, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp1, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp2, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_add_pd(Vtmp1, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_sub_pd(Vtmp1, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vtmp1, Vtmp2);)

ENABLE_AVX_IMPLEMENTATION(Vch = _mm256_add_pd(Vch, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = Vch;)
ENABLE_AVX_IMPLEMENTATION(Vch = _mm256_blendv_pd(Vsh, Vch, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_blendv_pd(Vtmp1, Vsh, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vch, Vch);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vsh, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp1, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_sqrt_pd(Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_div_pd(Vone, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Vtmp1, Vone_half);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp1, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp1, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp2, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_add_pd(Vtmp1, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_sub_pd(Vtmp1, Vtmp3);)

ENABLE_AVX_IMPLEMENTATION(Vch = _mm256_mul_pd(Vch, Vtmp1);)
ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_mul_pd(Vsh, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vc = _mm256_mul_pd(Vch, Vch);)
ENABLE_AVX_IMPLEMENTATION(Vs = _mm256_mul_pd(Vsh, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vc = _mm256_sub_pd(Vc, Vs);)
ENABLE_AVX_IMPLEMENTATION(Vs = _mm256_mul_pd(Vsh, Vch);)
ENABLE_AVX_IMPLEMENTATION(Vs = _mm256_add_pd(Vs, Vs);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vs, VA11);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vs, VA21);)
ENABLE_AVX_IMPLEMENTATION(VA11 = _mm256_mul_pd(Vc, VA11);)
ENABLE_AVX_IMPLEMENTATION(VA21 = _mm256_mul_pd(Vc, VA21);)
ENABLE_AVX_IMPLEMENTATION(VA11 = _mm256_add_pd(VA11, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(VA21 = _mm256_sub_pd(VA21, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vs, VA12);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vs, VA22);)
ENABLE_AVX_IMPLEMENTATION(VA12 = _mm256_mul_pd(Vc, VA12);)
ENABLE_AVX_IMPLEMENTATION(VA22 = _mm256_mul_pd(Vc, VA22);)
ENABLE_AVX_IMPLEMENTATION(VA12 = _mm256_add_pd(VA12, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(VA22 = _mm256_sub_pd(VA22, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vs, VA13);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vs, VA23);)
ENABLE_AVX_IMPLEMENTATION(VA13 = _mm256_mul_pd(Vc, VA13);)
ENABLE_AVX_IMPLEMENTATION(VA23 = _mm256_mul_pd(Vc, VA23);)
ENABLE_AVX_IMPLEMENTATION(VA13 = _mm256_add_pd(VA13, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(VA23 = _mm256_sub_pd(VA23, Vtmp1);)

//###########################################################
// Update matrix U
//###########################################################

#ifdef COMPUTE_U_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vs, VU11);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vs, VU12);)
ENABLE_AVX_IMPLEMENTATION(VU11 = _mm256_mul_pd(Vc, VU11);)
ENABLE_AVX_IMPLEMENTATION(VU12 = _mm256_mul_pd(Vc, VU12);)
ENABLE_AVX_IMPLEMENTATION(VU11 = _mm256_add_pd(VU11, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(VU12 = _mm256_sub_pd(VU12, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vs, VU21);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vs, VU22);)
ENABLE_AVX_IMPLEMENTATION(VU21 = _mm256_mul_pd(Vc, VU21);)
ENABLE_AVX_IMPLEMENTATION(VU22 = _mm256_mul_pd(Vc, VU22);)
ENABLE_AVX_IMPLEMENTATION(VU21 = _mm256_add_pd(VU21, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(VU22 = _mm256_sub_pd(VU22, Vtmp1);)

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vs, VU31);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vs, VU32);)
ENABLE_AVX_IMPLEMENTATION(VU31 = _mm256_mul_pd(Vc, VU31);)
ENABLE_AVX_IMPLEMENTATION(VU32 = _mm256_mul_pd(Vc, VU32);)
ENABLE_AVX_IMPLEMENTATION(VU31 = _mm256_add_pd(VU31, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(VU32 = _mm256_sub_pd(VU32, Vtmp1);)
#endif
