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
// Local variable declarations
//###########################################################

#ifdef PRINT_DEBUGGING_OUTPUT

#ifdef USE_SSE_IMPLEMENTATION
float buf[4];
float A11, A21, A31, A12, A22, A32, A13, A23, A33;
float S11, S21, S31, S22, S32, S33;
#ifdef COMPUTE_V_AS_QUATERNION
float QVS, QVVX, QVVY, QVVZ;
#endif
#ifdef COMPUTE_V_AS_MATRIX
float V11, V21, V31, V12, V22, V32, V13, V23, V33;
#endif
#ifdef COMPUTE_U_AS_QUATERNION
float QUS, QUVX, QUVY, QUVZ;
#endif
#ifdef COMPUTE_U_AS_MATRIX
float U11, U21, U31, U12, U22, U32, U13, U23, U33;
#endif
#endif

#ifdef USE_AVX_IMPLEMENTATION
double buf[4];
double A11, A21, A31, A12, A22, A32, A13, A23, A33;
double S11, S21, S31, S22, S32, S33;
#ifdef COMPUTE_V_AS_QUATERNION
double QVS, QVVX, QVVY, QVVZ;
#endif
#ifdef COMPUTE_V_AS_MATRIX
double V11, V21, V31, V12, V22, V32, V13, V23, V33;
#endif
#ifdef COMPUTE_U_AS_QUATERNION
double QUS, QUVX, QUVY, QUVZ;
#endif
#ifdef COMPUTE_U_AS_MATRIX
double U11, U21, U31, U12, U22, U32, U13, U23, U33;
#endif
#endif

#endif

const double Four_Gamma_Squared = sqrt(8.) + 3.;
const double Sine_Pi_Over_Eight = .5 * sqrt(2. - sqrt(2.));
const double Cosine_Pi_Over_Eight = .5 * sqrt(2. + sqrt(2.));

ENABLE_AVX_IMPLEMENTATION(__m256d Vfour_gamma_squared;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vsine_pi_over_eight;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vcosine_pi_over_eight;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vone_half;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vone;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vtiny_number;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vsmall_number;)

ENABLE_AVX_IMPLEMENTATION(Vfour_gamma_squared = _mm256_set1_pd(Four_Gamma_Squared);)
ENABLE_AVX_IMPLEMENTATION(Vsine_pi_over_eight = _mm256_set1_pd(Sine_Pi_Over_Eight);)
ENABLE_AVX_IMPLEMENTATION(Vcosine_pi_over_eight = _mm256_set1_pd(Cosine_Pi_Over_Eight);)
ENABLE_AVX_IMPLEMENTATION(Vone_half = _mm256_set1_pd(.5);)
ENABLE_AVX_IMPLEMENTATION(Vone = _mm256_set1_pd(1.);)
ENABLE_AVX_IMPLEMENTATION(Vtiny_number = _mm256_set1_pd(1.e-20);)
ENABLE_AVX_IMPLEMENTATION(Vsmall_number = _mm256_set1_pd(1.e-12);)

ENABLE_AVX_IMPLEMENTATION(__m256d Va11;)
ENABLE_AVX_IMPLEMENTATION(__m256d Va21;)
ENABLE_AVX_IMPLEMENTATION(__m256d Va31;)
ENABLE_AVX_IMPLEMENTATION(__m256d Va12;)
ENABLE_AVX_IMPLEMENTATION(__m256d Va22;)
ENABLE_AVX_IMPLEMENTATION(__m256d Va32;)
ENABLE_AVX_IMPLEMENTATION(__m256d Va13;)
ENABLE_AVX_IMPLEMENTATION(__m256d Va23;)
ENABLE_AVX_IMPLEMENTATION(__m256d Va33;)

#ifdef COMPUTE_V_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(__m256d Vv11;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vv21;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vv31;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vv12;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vv22;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vv32;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vv13;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vv23;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vv33;)
#endif

#ifdef COMPUTE_V_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(__m256d Vqvs;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vqvvx;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vqvvy;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vqvvz;)
#endif

#ifdef COMPUTE_U_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(__m256d Vu11;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vu21;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vu31;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vu12;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vu22;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vu32;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vu13;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vu23;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vu33;)
#endif

#ifdef COMPUTE_U_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(__m256d Vqus;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vquvx;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vquvy;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vquvz;)
#endif

ENABLE_AVX_IMPLEMENTATION(__m256d Vc;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vs;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vch;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vsh;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vtmp1;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vtmp2;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vtmp3;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vtmp4;)
ENABLE_AVX_IMPLEMENTATION(__m256d Vtmp5;)
