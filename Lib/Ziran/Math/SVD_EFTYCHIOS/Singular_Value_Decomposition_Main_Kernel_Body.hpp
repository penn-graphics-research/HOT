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

#ifdef __INTEL_COMPILER
#pragma warning(disable : 592)
#endif

// #define USE_ACCURATE_RSQRT_IN_JACOBI_CONJUGATION
// #define PERFORM_STRICT_QUATERNION_RENORMALIZATION

{ // Begin block : Scope of qV (if not maintained)

#ifndef COMPUTE_V_AS_QUATERNION
    ENABLE_AVX_IMPLEMENTATION(__m256d Vqvs;)
    ENABLE_AVX_IMPLEMENTATION(__m256d Vqvvx;)
    ENABLE_AVX_IMPLEMENTATION(__m256d Vqvvy;)
    ENABLE_AVX_IMPLEMENTATION(__m256d Vqvvz;)
#endif

    { // Begin block : Symmetric eigenanalysis

        ENABLE_AVX_IMPLEMENTATION(__m256d Vs11;)
        ENABLE_AVX_IMPLEMENTATION(__m256d Vs21;)
        ENABLE_AVX_IMPLEMENTATION(__m256d Vs31;)
        ENABLE_AVX_IMPLEMENTATION(__m256d Vs22;)
        ENABLE_AVX_IMPLEMENTATION(__m256d Vs32;)
        ENABLE_AVX_IMPLEMENTATION(__m256d Vs33;)

        ENABLE_AVX_IMPLEMENTATION(Vqvs = Vone;)
        ENABLE_AVX_IMPLEMENTATION(Vqvvx = _mm256_xor_pd(Vqvvx, Vqvvx);)
        ENABLE_AVX_IMPLEMENTATION(Vqvvy = _mm256_xor_pd(Vqvvy, Vqvvy);)
        ENABLE_AVX_IMPLEMENTATION(Vqvvz = _mm256_xor_pd(Vqvvz, Vqvvz);)

        //###########################################################
        // Compute normal equations matrix
        //###########################################################

        ENABLE_AVX_IMPLEMENTATION(Vs11 = _mm256_mul_pd(Va11, Va11);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va21, Va21);)
        ENABLE_AVX_IMPLEMENTATION(Vs11 = _mm256_add_pd(Vtmp1, Vs11);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va31, Va31);)
        ENABLE_AVX_IMPLEMENTATION(Vs11 = _mm256_add_pd(Vtmp1, Vs11);)

        ENABLE_AVX_IMPLEMENTATION(Vs21 = _mm256_mul_pd(Va12, Va11);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va22, Va21);)
        ENABLE_AVX_IMPLEMENTATION(Vs21 = _mm256_add_pd(Vtmp1, Vs21);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va32, Va31);)
        ENABLE_AVX_IMPLEMENTATION(Vs21 = _mm256_add_pd(Vtmp1, Vs21);)

        ENABLE_AVX_IMPLEMENTATION(Vs31 = _mm256_mul_pd(Va13, Va11);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va23, Va21);)
        ENABLE_AVX_IMPLEMENTATION(Vs31 = _mm256_add_pd(Vtmp1, Vs31);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va33, Va31);)
        ENABLE_AVX_IMPLEMENTATION(Vs31 = _mm256_add_pd(Vtmp1, Vs31);)

        ENABLE_AVX_IMPLEMENTATION(Vs22 = _mm256_mul_pd(Va12, Va12);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va22, Va22);)
        ENABLE_AVX_IMPLEMENTATION(Vs22 = _mm256_add_pd(Vtmp1, Vs22);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va32, Va32);)
        ENABLE_AVX_IMPLEMENTATION(Vs22 = _mm256_add_pd(Vtmp1, Vs22);)

        ENABLE_AVX_IMPLEMENTATION(Vs32 = _mm256_mul_pd(Va13, Va12);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va23, Va22);)
        ENABLE_AVX_IMPLEMENTATION(Vs32 = _mm256_add_pd(Vtmp1, Vs32);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va33, Va32);)
        ENABLE_AVX_IMPLEMENTATION(Vs32 = _mm256_add_pd(Vtmp1, Vs32);)

        ENABLE_AVX_IMPLEMENTATION(Vs33 = _mm256_mul_pd(Va13, Va13);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va23, Va23);)
        ENABLE_AVX_IMPLEMENTATION(Vs33 = _mm256_add_pd(Vtmp1, Vs33);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va33, Va33);)
        ENABLE_AVX_IMPLEMENTATION(Vs33 = _mm256_add_pd(Vtmp1, Vs33);)

        //###########################################################
        // Solve symmetric eigenproblem using Jacobi iteration
        //###########################################################

        for (int sweep = 1; sweep <= 4; sweep++) {

            // First Jacobi conjugation

#define SS11 Ss11
#define SS21 Ss21
#define SS31 Ss31
#define SS22 Ss22
#define SS32 Ss32
#define SS33 Ss33
#define SQVVX Sqvvx
#define SQVVY Sqvvy
#define SQVVZ Sqvvz
#define STMP1 Stmp1
#define STMP2 Stmp2
#define STMP3 Stmp3

#define VS11 Vs11
#define VS21 Vs21
#define VS31 Vs31
#define VS22 Vs22
#define VS32 Vs32
#define VS33 Vs33
#define VQVVX Vqvvx
#define VQVVY Vqvvy
#define VQVVZ Vqvvz
#define VTMP1 Vtmp1
#define VTMP2 Vtmp2
#define VTMP3 Vtmp3

#include "Singular_Value_Decomposition_Jacobi_Conjugation_Kernel.hpp"

#undef SS11
#undef SS21
#undef SS31
#undef SS22
#undef SS32
#undef SS33
#undef SQVVX
#undef SQVVY
#undef SQVVZ
#undef STMP1
#undef STMP2
#undef STMP3

#undef VS11
#undef VS21
#undef VS31
#undef VS22
#undef VS32
#undef VS33
#undef VQVVX
#undef VQVVY
#undef VQVVZ
#undef VTMP1
#undef VTMP2
#undef VTMP3

            // Second Jacobi conjugation

#define SS11 Ss22
#define SS21 Ss32
#define SS31 Ss21
#define SS22 Ss33
#define SS32 Ss31
#define SS33 Ss11
#define SQVVX Sqvvy
#define SQVVY Sqvvz
#define SQVVZ Sqvvx
#define STMP1 Stmp2
#define STMP2 Stmp3
#define STMP3 Stmp1

#define VS11 Vs22
#define VS21 Vs32
#define VS31 Vs21
#define VS22 Vs33
#define VS32 Vs31
#define VS33 Vs11
#define VQVVX Vqvvy
#define VQVVY Vqvvz
#define VQVVZ Vqvvx
#define VTMP1 Vtmp2
#define VTMP2 Vtmp3
#define VTMP3 Vtmp1

#include "Singular_Value_Decomposition_Jacobi_Conjugation_Kernel.hpp"

#undef SS11
#undef SS21
#undef SS31
#undef SS22
#undef SS32
#undef SS33
#undef SQVVX
#undef SQVVY
#undef SQVVZ
#undef STMP1
#undef STMP2
#undef STMP3

#undef VS11
#undef VS21
#undef VS31
#undef VS22
#undef VS32
#undef VS33
#undef VQVVX
#undef VQVVY
#undef VQVVZ
#undef VTMP1
#undef VTMP2
#undef VTMP3

            // Third Jacobi conjugation

#define SS11 Ss33
#define SS21 Ss31
#define SS31 Ss32
#define SS22 Ss11
#define SS32 Ss21
#define SS33 Ss22
#define SQVVX Sqvvz
#define SQVVY Sqvvx
#define SQVVZ Sqvvy
#define STMP1 Stmp3
#define STMP2 Stmp1
#define STMP3 Stmp2

#define VS11 Vs33
#define VS21 Vs31
#define VS31 Vs32
#define VS22 Vs11
#define VS32 Vs21
#define VS33 Vs22
#define VQVVX Vqvvz
#define VQVVY Vqvvx
#define VQVVZ Vqvvy
#define VTMP1 Vtmp3
#define VTMP2 Vtmp1
#define VTMP3 Vtmp2

#include "Singular_Value_Decomposition_Jacobi_Conjugation_Kernel.hpp"

#undef SS11
#undef SS21
#undef SS31
#undef SS22
#undef SS32
#undef SS33
#undef SQVVX
#undef SQVVY
#undef SQVVZ
#undef STMP1
#undef STMP2
#undef STMP3

#undef VS11
#undef VS21
#undef VS31
#undef VS22
#undef VS32
#undef VS33
#undef VQVVX
#undef VQVVY
#undef VQVVZ
#undef VTMP1
#undef VTMP2
#undef VTMP3
        }

    } // End block : Symmetric eigenanalysis

    //###########################################################
    // Normalize quaternion for matrix V
    //###########################################################

#if !defined(USE_ACCURATE_RSQRT_IN_JACOBI_CONJUGATION) || defined(PERFORM_STRICT_QUATERNION_RENORMALIZATION)

    ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vqvs, Vqvs);)
    ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vqvvx, Vqvvx);)
    ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp1, Vtmp2);)
    ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vqvvy, Vqvvy);)
    ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp1, Vtmp2);)
    ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vqvvz, Vqvvz);)
    ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp1, Vtmp2);)

    /*ENABLE_AVX_IMPLEMENTATION(Vtmp1=_mm256_rsqrt_pd(Vtmp2);)*/ ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_sqrt_pd(Vtmp2);) ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_div_pd(Vone, Vtmp1);)
            ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Vtmp1, Vone_half);)
                ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp1, Vtmp4);)
                    ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp1, Vtmp3);)
                        ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp2, Vtmp3);)
                            ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_add_pd(Vtmp1, Vtmp4);)
                                ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_sub_pd(Vtmp1, Vtmp3);)

                                    ENABLE_AVX_IMPLEMENTATION(Vqvs = _mm256_mul_pd(Vqvs, Vtmp1);)
                                        ENABLE_AVX_IMPLEMENTATION(Vqvvx = _mm256_mul_pd(Vqvvx, Vtmp1);)
                                            ENABLE_AVX_IMPLEMENTATION(Vqvvy = _mm256_mul_pd(Vqvvy, Vtmp1);)
                                                ENABLE_AVX_IMPLEMENTATION(Vqvvz = _mm256_mul_pd(Vqvvz, Vtmp1);)

#ifdef PRINT_DEBUGGING_OUTPUT
#ifdef USE_SCALAR_IMPLEMENTATION
                                                    std::cout
        << "Scalar qV =" << std::endl;
    std::cout << std::setw(12) << Sqvs.f << "  " << std::setw(12) << Sqvvx.f << "  " << std::setw(12) << Sqvvy.f << "  " << std::setw(12) << Sqvvz.f << std::endl;
#endif
#ifdef USE_SSE_IMPLEMENTATION
    _mm_storeu_ps(buf, Vqvs);
    QVS = buf[0];
    _mm_storeu_ps(buf, Vqvvx);
    QVVX = buf[0];
    _mm_storeu_ps(buf, Vqvvy);
    QVVY = buf[0];
    _mm_storeu_ps(buf, Vqvvz);
    QVVZ = buf[0];
    std::cout << "Vector qV =" << std::endl;
    std::cout << std::setw(12) << QVS << "  " << std::setw(12) << QVVX << "  " << std::setw(12) << QVVY << "  " << std::setw(12) << QVVZ << std::endl;
#endif
#ifdef USE_AVX_IMPLEMENTATION
    _mm256_storeu_pd(buf, Vqvs);
    QVS = buf[0];
    _mm256_storeu_pd(buf, Vqvvx);
    QVVX = buf[0];
    _mm256_storeu_pd(buf, Vqvvy);
    QVVY = buf[0];
    _mm256_storeu_pd(buf, Vqvvz);
    QVVZ = buf[0];
    std::cout << "Vector qV =" << std::endl;
    std::cout << std::setw(12) << QVS << "  " << std::setw(12) << QVVX << "  " << std::setw(12) << QVVY << "  " << std::setw(12) << QVVZ << std::endl;
#endif
#endif

#endif

    { // Begin block : Conjugation with V

        //###########################################################
        // Transform quaternion to matrix V
        //###########################################################

#ifndef COMPUTE_V_AS_MATRIX

#endif

        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vqvvx, Vqvvx);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vqvvy, Vqvvy);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vqvvz, Vqvvz);)
        ENABLE_AVX_IMPLEMENTATION(Vv11 = _mm256_mul_pd(Vqvs, Vqvs);)
        ENABLE_AVX_IMPLEMENTATION(Vv22 = _mm256_sub_pd(Vv11, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vv33 = _mm256_sub_pd(Vv22, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Vv33 = _mm256_add_pd(Vv33, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Vv22 = _mm256_add_pd(Vv22, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Vv22 = _mm256_sub_pd(Vv22, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Vv11 = _mm256_add_pd(Vv11, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vv11 = _mm256_sub_pd(Vv11, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Vv11 = _mm256_sub_pd(Vv11, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_add_pd(Vqvvx, Vqvvx);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vqvvy, Vqvvy);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_add_pd(Vqvvz, Vqvvz);)
        ENABLE_AVX_IMPLEMENTATION(Vv32 = _mm256_mul_pd(Vqvs, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vv13 = _mm256_mul_pd(Vqvs, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Vv21 = _mm256_mul_pd(Vqvs, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vqvvy, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vqvvz, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vqvvx, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Vv12 = _mm256_sub_pd(Vtmp1, Vv21);)
        ENABLE_AVX_IMPLEMENTATION(Vv23 = _mm256_sub_pd(Vtmp2, Vv32);)
        ENABLE_AVX_IMPLEMENTATION(Vv31 = _mm256_sub_pd(Vtmp3, Vv13);)
        ENABLE_AVX_IMPLEMENTATION(Vv21 = _mm256_add_pd(Vtmp1, Vv21);)
        ENABLE_AVX_IMPLEMENTATION(Vv32 = _mm256_add_pd(Vtmp2, Vv32);)
        ENABLE_AVX_IMPLEMENTATION(Vv13 = _mm256_add_pd(Vtmp3, Vv13);)

#ifdef COMPUTE_V_AS_MATRIX
#ifdef PRINT_DEBUGGING_OUTPUT
#ifdef USE_SCALAR_IMPLEMENTATION
        std::cout << "Scalar V =" << std::endl;
        std::cout << std::setw(12) << Sv11.f << "  " << std::setw(12) << Sv12.f << "  " << std::setw(12) << Sv13.f << std::endl;
        std::cout << std::setw(12) << Sv21.f << "  " << std::setw(12) << Sv22.f << "  " << std::setw(12) << Sv23.f << std::endl;
        std::cout << std::setw(12) << Sv31.f << "  " << std::setw(12) << Sv32.f << "  " << std::setw(12) << Sv33.f << std::endl;
#endif
#ifdef USE_SSE_IMPLEMENTATION
        _mm_storeu_ps(buf, Vv11);
        V11 = buf[0];
        _mm_storeu_ps(buf, Vv21);
        V21 = buf[0];
        _mm_storeu_ps(buf, Vv31);
        V31 = buf[0];
        _mm_storeu_ps(buf, Vv12);
        V12 = buf[0];
        _mm_storeu_ps(buf, Vv22);
        V22 = buf[0];
        _mm_storeu_ps(buf, Vv32);
        V32 = buf[0];
        _mm_storeu_ps(buf, Vv13);
        V13 = buf[0];
        _mm_storeu_ps(buf, Vv23);
        V23 = buf[0];
        _mm_storeu_ps(buf, Vv33);
        V33 = buf[0];
        std::cout << "Vector V =" << std::endl;
        std::cout << std::setw(12) << V11 << "  " << std::setw(12) << V12 << "  " << std::setw(12) << V13 << std::endl;
        std::cout << std::setw(12) << V21 << "  " << std::setw(12) << V22 << "  " << std::setw(12) << V23 << std::endl;
        std::cout << std::setw(12) << V31 << "  " << std::setw(12) << V32 << "  " << std::setw(12) << V33 << std::endl;
#endif
#ifdef USE_AVX_IMPLEMENTATION
        _mm256_storeu_pd(buf, Vv11);
        V11 = buf[0];
        _mm256_storeu_pd(buf, Vv21);
        V21 = buf[0];
        _mm256_storeu_pd(buf, Vv31);
        V31 = buf[0];
        _mm256_storeu_pd(buf, Vv12);
        V12 = buf[0];
        _mm256_storeu_pd(buf, Vv22);
        V22 = buf[0];
        _mm256_storeu_pd(buf, Vv32);
        V32 = buf[0];
        _mm256_storeu_pd(buf, Vv13);
        V13 = buf[0];
        _mm256_storeu_pd(buf, Vv23);
        V23 = buf[0];
        _mm256_storeu_pd(buf, Vv33);
        V33 = buf[0];
        std::cout << "Vector V =" << std::endl;
        std::cout << std::setw(12) << V11 << "  " << std::setw(12) << V12 << "  " << std::setw(12) << V13 << std::endl;
        std::cout << std::setw(12) << V21 << "  " << std::setw(12) << V22 << "  " << std::setw(12) << V23 << std::endl;
        std::cout << std::setw(12) << V31 << "  " << std::setw(12) << V32 << "  " << std::setw(12) << V33 << std::endl;
#endif
#endif
#endif

        //###########################################################
        // Multiply (from the right) with V
        //###########################################################

        ENABLE_AVX_IMPLEMENTATION(Vtmp2 = Va12;)
        ENABLE_AVX_IMPLEMENTATION(Vtmp3 = Va13;)
        ENABLE_AVX_IMPLEMENTATION(Va12 = _mm256_mul_pd(Vv12, Va11);)
        ENABLE_AVX_IMPLEMENTATION(Va13 = _mm256_mul_pd(Vv13, Va11);)
        ENABLE_AVX_IMPLEMENTATION(Va11 = _mm256_mul_pd(Vv11, Va11);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv21, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Va11 = _mm256_add_pd(Va11, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv31, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Va11 = _mm256_add_pd(Va11, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv22, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Va12 = _mm256_add_pd(Va12, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv32, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Va12 = _mm256_add_pd(Va12, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv23, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Va13 = _mm256_add_pd(Va13, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv33, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Va13 = _mm256_add_pd(Va13, Vtmp1);)

        ENABLE_AVX_IMPLEMENTATION(Vtmp2 = Va22;)
        ENABLE_AVX_IMPLEMENTATION(Vtmp3 = Va23;)
        ENABLE_AVX_IMPLEMENTATION(Va22 = _mm256_mul_pd(Vv12, Va21);)
        ENABLE_AVX_IMPLEMENTATION(Va23 = _mm256_mul_pd(Vv13, Va21);)
        ENABLE_AVX_IMPLEMENTATION(Va21 = _mm256_mul_pd(Vv11, Va21);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv21, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Va21 = _mm256_add_pd(Va21, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv31, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Va21 = _mm256_add_pd(Va21, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv22, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Va22 = _mm256_add_pd(Va22, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv32, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Va22 = _mm256_add_pd(Va22, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv23, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Va23 = _mm256_add_pd(Va23, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv33, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Va23 = _mm256_add_pd(Va23, Vtmp1);)

        ENABLE_AVX_IMPLEMENTATION(Vtmp2 = Va32;)
        ENABLE_AVX_IMPLEMENTATION(Vtmp3 = Va33;)
        ENABLE_AVX_IMPLEMENTATION(Va32 = _mm256_mul_pd(Vv12, Va31);)
        ENABLE_AVX_IMPLEMENTATION(Va33 = _mm256_mul_pd(Vv13, Va31);)
        ENABLE_AVX_IMPLEMENTATION(Va31 = _mm256_mul_pd(Vv11, Va31);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv21, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Va31 = _mm256_add_pd(Va31, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv31, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Va31 = _mm256_add_pd(Va31, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv22, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Va32 = _mm256_add_pd(Va32, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv32, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Va32 = _mm256_add_pd(Va32, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv23, Vtmp2);)
        ENABLE_AVX_IMPLEMENTATION(Va33 = _mm256_add_pd(Va33, Vtmp1);)
        ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vv33, Vtmp3);)
        ENABLE_AVX_IMPLEMENTATION(Va33 = _mm256_add_pd(Va33, Vtmp1);)

#ifdef PRINT_DEBUGGING_OUTPUT
#ifdef USE_SCALAR_IMPLEMENTATION
        std::cout << "Scalar A (after multiplying with V) =" << std::endl;
        std::cout << std::setw(12) << Sa11.f << "  " << std::setw(12) << Sa12.f << "  " << std::setw(12) << Sa13.f << std::endl;
        std::cout << std::setw(12) << Sa21.f << "  " << std::setw(12) << Sa22.f << "  " << std::setw(12) << Sa23.f << std::endl;
        std::cout << std::setw(12) << Sa31.f << "  " << std::setw(12) << Sa32.f << "  " << std::setw(12) << Sa33.f << std::endl;
#endif
#ifdef USE_SSE_IMPLEMENTATION
        _mm_storeu_ps(buf, Va11);
        A11 = buf[0];
        _mm_storeu_ps(buf, Va21);
        A21 = buf[0];
        _mm_storeu_ps(buf, Va31);
        A31 = buf[0];
        _mm_storeu_ps(buf, Va12);
        A12 = buf[0];
        _mm_storeu_ps(buf, Va22);
        A22 = buf[0];
        _mm_storeu_ps(buf, Va32);
        A32 = buf[0];
        _mm_storeu_ps(buf, Va13);
        A13 = buf[0];
        _mm_storeu_ps(buf, Va23);
        A23 = buf[0];
        _mm_storeu_ps(buf, Va33);
        A33 = buf[0];
        std::cout << "Vector A (after multiplying with V) =" << std::endl;
        std::cout << std::setw(12) << A11 << "  " << std::setw(12) << A12 << "  " << std::setw(12) << A13 << std::endl;
        std::cout << std::setw(12) << A21 << "  " << std::setw(12) << A22 << "  " << std::setw(12) << A23 << std::endl;
        std::cout << std::setw(12) << A31 << "  " << std::setw(12) << A32 << "  " << std::setw(12) << A33 << std::endl;
#endif
#ifdef USE_AVX_IMPLEMENTATION
        _mm256_storeu_pd(buf, Va11);
        A11 = buf[0];
        _mm256_storeu_pd(buf, Va21);
        A21 = buf[0];
        _mm256_storeu_pd(buf, Va31);
        A31 = buf[0];
        _mm256_storeu_pd(buf, Va12);
        A12 = buf[0];
        _mm256_storeu_pd(buf, Va22);
        A22 = buf[0];
        _mm256_storeu_pd(buf, Va32);
        A32 = buf[0];
        _mm256_storeu_pd(buf, Va13);
        A13 = buf[0];
        _mm256_storeu_pd(buf, Va23);
        A23 = buf[0];
        _mm256_storeu_pd(buf, Va33);
        A33 = buf[0];
        std::cout << "Vector A (after multiplying with V) =" << std::endl;
        std::cout << std::setw(12) << A11 << "  " << std::setw(12) << A12 << "  " << std::setw(12) << A13 << std::endl;
        std::cout << std::setw(12) << A21 << "  " << std::setw(12) << A22 << "  " << std::setw(12) << A23 << std::endl;
        std::cout << std::setw(12) << A31 << "  " << std::setw(12) << A32 << "  " << std::setw(12) << A33 << std::endl;
#endif
#endif

    } // End block : Conjugation with V

} // End block : Scope of qV (if not maintained)

//###########################################################
// Permute columns such that the singular values are sorted
//###########################################################

ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Va11, Va11);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Va21, Va21);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_add_pd(Vtmp1, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Va31, Va31);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_add_pd(Vtmp1, Vtmp4);)

ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Va12, Va12);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Va22, Va22);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp2, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Va32, Va32);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp2, Vtmp4);)

ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Va13, Va13);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Va23, Va23);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_add_pd(Vtmp3, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Va33, Va33);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_add_pd(Vtmp3, Vtmp4);)

ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_cmp_pd(Vtmp1, Vtmp2, _CMP_LT_OS);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Va11, Va12);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va11 = _mm256_xor_pd(Va11, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Va12 = _mm256_xor_pd(Va12, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Va21, Va22);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va21 = _mm256_xor_pd(Va21, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Va22 = _mm256_xor_pd(Va22, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Va31, Va32);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va31 = _mm256_xor_pd(Va31, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Va32 = _mm256_xor_pd(Va32, Vtmp5);)

#ifdef COMPUTE_V_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vv11, Vv12);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv11 = _mm256_xor_pd(Vv11, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vv12 = _mm256_xor_pd(Vv12, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vv21, Vv22);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv21 = _mm256_xor_pd(Vv21, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vv22 = _mm256_xor_pd(Vv22, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vv31, Vv32);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv31 = _mm256_xor_pd(Vv31, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vv32 = _mm256_xor_pd(Vv32, Vtmp5);)
#endif

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vtmp1, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_xor_pd(Vtmp1, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_xor_pd(Vtmp2, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_set1_pd(-2.);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = Vone;)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_add_pd(Vtmp4, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Va12 = _mm256_mul_pd(Va12, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va22 = _mm256_mul_pd(Va22, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va32 = _mm256_mul_pd(Va32, Vtmp4);)

#ifdef COMPUTE_V_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(Vv12 = _mm256_mul_pd(Vv12, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv22 = _mm256_mul_pd(Vv22, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv32 = _mm256_mul_pd(Vv32, Vtmp4);)
#endif

// If columns 1-2 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)

#ifdef COMPUTE_V_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Vtmp4, Vone_half);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_sub_pd(Vtmp4, Vone_half);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_mul_pd(Vtmp4, Vqvvz);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_add_pd(Vtmp5, Vqvs);)
ENABLE_AVX_IMPLEMENTATION(Vqvs = _mm256_mul_pd(Vqvs, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vqvvz = _mm256_sub_pd(Vqvvz, Vqvs);)
ENABLE_AVX_IMPLEMENTATION(Vqvs = Vtmp5;)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_mul_pd(Vtmp4, Vqvvx);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_add_pd(Vtmp5, Vqvvy);)
ENABLE_AVX_IMPLEMENTATION(Vqvvy = _mm256_mul_pd(Vqvvy, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vqvvx = _mm256_sub_pd(Vqvvx, Vqvvy);)
ENABLE_AVX_IMPLEMENTATION(Vqvvy = Vtmp5;)
#endif

// Swap columns 1-3 if necessary

ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_cmp_pd(Vtmp1, Vtmp3, _CMP_LT_OS);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Va11, Va13);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va11 = _mm256_xor_pd(Va11, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Va13 = _mm256_xor_pd(Va13, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Va21, Va23);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va21 = _mm256_xor_pd(Va21, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Va23 = _mm256_xor_pd(Va23, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Va31, Va33);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va31 = _mm256_xor_pd(Va31, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Va33 = _mm256_xor_pd(Va33, Vtmp5);)

#ifdef COMPUTE_V_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vv11, Vv13);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv11 = _mm256_xor_pd(Vv11, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vv13 = _mm256_xor_pd(Vv13, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vv21, Vv23);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv21 = _mm256_xor_pd(Vv21, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vv23 = _mm256_xor_pd(Vv23, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vv31, Vv33);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv31 = _mm256_xor_pd(Vv31, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vv33 = _mm256_xor_pd(Vv33, Vtmp5);)
#endif

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vtmp1, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_xor_pd(Vtmp1, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_xor_pd(Vtmp3, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_set1_pd(-2.);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = Vone;)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_add_pd(Vtmp4, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Va11 = _mm256_mul_pd(Va11, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va21 = _mm256_mul_pd(Va21, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va31 = _mm256_mul_pd(Va31, Vtmp4);)

#ifdef COMPUTE_V_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(Vv11 = _mm256_mul_pd(Vv11, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv21 = _mm256_mul_pd(Vv21, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv31 = _mm256_mul_pd(Vv31, Vtmp4);)
#endif

// If columns 1-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)

#ifdef COMPUTE_V_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Vtmp4, Vone_half);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_sub_pd(Vtmp4, Vone_half);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_mul_pd(Vtmp4, Vqvvy);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_add_pd(Vtmp5, Vqvs);)
ENABLE_AVX_IMPLEMENTATION(Vqvs = _mm256_mul_pd(Vqvs, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vqvvy = _mm256_sub_pd(Vqvvy, Vqvs);)
ENABLE_AVX_IMPLEMENTATION(Vqvs = Vtmp5;)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_mul_pd(Vtmp4, Vqvvz);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_add_pd(Vtmp5, Vqvvx);)
ENABLE_AVX_IMPLEMENTATION(Vqvvx = _mm256_mul_pd(Vqvvx, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vqvvz = _mm256_sub_pd(Vqvvz, Vqvvx);)
ENABLE_AVX_IMPLEMENTATION(Vqvvx = Vtmp5;)
#endif

// Swap columns 2-3 if necessary

ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_cmp_pd(Vtmp2, Vtmp3, _CMP_LT_OS);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Va12, Va13);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va12 = _mm256_xor_pd(Va12, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Va13 = _mm256_xor_pd(Va13, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Va22, Va23);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va22 = _mm256_xor_pd(Va22, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Va23 = _mm256_xor_pd(Va23, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Va32, Va33);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va32 = _mm256_xor_pd(Va32, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Va33 = _mm256_xor_pd(Va33, Vtmp5);)

#ifdef COMPUTE_V_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vv12, Vv13);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv12 = _mm256_xor_pd(Vv12, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vv13 = _mm256_xor_pd(Vv13, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vv22, Vv23);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv22 = _mm256_xor_pd(Vv22, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vv23 = _mm256_xor_pd(Vv23, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vv32, Vv33);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv32 = _mm256_xor_pd(Vv32, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vv33 = _mm256_xor_pd(Vv33, Vtmp5);)
#endif

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_xor_pd(Vtmp2, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_xor_pd(Vtmp2, Vtmp5);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_xor_pd(Vtmp3, Vtmp5);)

// If columns 2-3 have been swapped, negate 3rd column of A and V so that V is still a rotation

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_set1_pd(-2.);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_and_pd(Vtmp5, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = Vone;)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_add_pd(Vtmp4, Vtmp5);)

ENABLE_AVX_IMPLEMENTATION(Va13 = _mm256_mul_pd(Va13, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va23 = _mm256_mul_pd(Va23, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Va33 = _mm256_mul_pd(Va33, Vtmp4);)

#ifdef COMPUTE_V_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(Vv13 = _mm256_mul_pd(Vv13, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv23 = _mm256_mul_pd(Vv23, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vv33 = _mm256_mul_pd(Vv33, Vtmp4);)
#endif

// If columns 2-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)

#ifdef COMPUTE_V_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Vtmp4, Vone_half);)
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_sub_pd(Vtmp4, Vone_half);)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_mul_pd(Vtmp4, Vqvvx);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_add_pd(Vtmp5, Vqvs);)
ENABLE_AVX_IMPLEMENTATION(Vqvs = _mm256_mul_pd(Vqvs, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vqvvx = _mm256_sub_pd(Vqvvx, Vqvs);)
ENABLE_AVX_IMPLEMENTATION(Vqvs = Vtmp5;)

ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_mul_pd(Vtmp4, Vqvvy);)
ENABLE_AVX_IMPLEMENTATION(Vtmp5 = _mm256_add_pd(Vtmp5, Vqvvz);)
ENABLE_AVX_IMPLEMENTATION(Vqvvz = _mm256_mul_pd(Vqvvz, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vqvvy = _mm256_sub_pd(Vqvvy, Vqvvz);)
ENABLE_AVX_IMPLEMENTATION(Vqvvz = Vtmp5;)
#endif

#ifdef COMPUTE_V_AS_MATRIX
#ifdef PRINT_DEBUGGING_OUTPUT
#ifdef USE_SCALAR_IMPLEMENTATION
std::cout << "Scalar V =" << std::endl;
std::cout << std::setw(12) << Sv11.f << "  " << std::setw(12) << Sv12.f << "  " << std::setw(12) << Sv13.f << std::endl;
std::cout << std::setw(12) << Sv21.f << "  " << std::setw(12) << Sv22.f << "  " << std::setw(12) << Sv23.f << std::endl;
std::cout << std::setw(12) << Sv31.f << "  " << std::setw(12) << Sv32.f << "  " << std::setw(12) << Sv33.f << std::endl;
#endif
#ifdef USE_SSE_IMPLEMENTATION
_mm_storeu_ps(buf, Vv11);
V11 = buf[0];
_mm_storeu_ps(buf, Vv21);
V21 = buf[0];
_mm_storeu_ps(buf, Vv31);
V31 = buf[0];
_mm_storeu_ps(buf, Vv12);
V12 = buf[0];
_mm_storeu_ps(buf, Vv22);
V22 = buf[0];
_mm_storeu_ps(buf, Vv32);
V32 = buf[0];
_mm_storeu_ps(buf, Vv13);
V13 = buf[0];
_mm_storeu_ps(buf, Vv23);
V23 = buf[0];
_mm_storeu_ps(buf, Vv33);
V33 = buf[0];
std::cout << "Vector V =" << std::endl;
std::cout << std::setw(12) << V11 << "  " << std::setw(12) << V12 << "  " << std::setw(12) << V13 << std::endl;
std::cout << std::setw(12) << V21 << "  " << std::setw(12) << V22 << "  " << std::setw(12) << V23 << std::endl;
std::cout << std::setw(12) << V31 << "  " << std::setw(12) << V32 << "  " << std::setw(12) << V33 << std::endl;
#endif
#ifdef USE_AVX_IMPLEMENTATION
_mm256_storeu_pd(buf, Vv11);
V11 = buf[0];
_mm256_storeu_pd(buf, Vv21);
V21 = buf[0];
_mm256_storeu_pd(buf, Vv31);
V31 = buf[0];
_mm256_storeu_pd(buf, Vv12);
V12 = buf[0];
_mm256_storeu_pd(buf, Vv22);
V22 = buf[0];
_mm256_storeu_pd(buf, Vv32);
V32 = buf[0];
_mm256_storeu_pd(buf, Vv13);
V13 = buf[0];
_mm256_storeu_pd(buf, Vv23);
V23 = buf[0];
_mm256_storeu_pd(buf, Vv33);
V33 = buf[0];
std::cout << "Vector V =" << std::endl;
std::cout << std::setw(12) << V11 << "  " << std::setw(12) << V12 << "  " << std::setw(12) << V13 << std::endl;
std::cout << std::setw(12) << V21 << "  " << std::setw(12) << V22 << "  " << std::setw(12) << V23 << std::endl;
std::cout << std::setw(12) << V31 << "  " << std::setw(12) << V32 << "  " << std::setw(12) << V33 << std::endl;
#endif
#endif
#endif

#ifdef PRINT_DEBUGGING_OUTPUT
#ifdef USE_SCALAR_IMPLEMENTATION
std::cout << "Scalar A (after multiplying with V) =" << std::endl;
std::cout << std::setw(12) << Sa11.f << "  " << std::setw(12) << Sa12.f << "  " << std::setw(12) << Sa13.f << std::endl;
std::cout << std::setw(12) << Sa21.f << "  " << std::setw(12) << Sa22.f << "  " << std::setw(12) << Sa23.f << std::endl;
std::cout << std::setw(12) << Sa31.f << "  " << std::setw(12) << Sa32.f << "  " << std::setw(12) << Sa33.f << std::endl;
#endif
#ifdef USE_SSE_IMPLEMENTATION
_mm_storeu_ps(buf, Va11);
A11 = buf[0];
_mm_storeu_ps(buf, Va21);
A21 = buf[0];
_mm_storeu_ps(buf, Va31);
A31 = buf[0];
_mm_storeu_ps(buf, Va12);
A12 = buf[0];
_mm_storeu_ps(buf, Va22);
A22 = buf[0];
_mm_storeu_ps(buf, Va32);
A32 = buf[0];
_mm_storeu_ps(buf, Va13);
A13 = buf[0];
_mm_storeu_ps(buf, Va23);
A23 = buf[0];
_mm_storeu_ps(buf, Va33);
A33 = buf[0];
std::cout << "Vector A (after multiplying with V) =" << std::endl;
std::cout << std::setw(12) << A11 << "  " << std::setw(12) << A12 << "  " << std::setw(12) << A13 << std::endl;
std::cout << std::setw(12) << A21 << "  " << std::setw(12) << A22 << "  " << std::setw(12) << A23 << std::endl;
std::cout << std::setw(12) << A31 << "  " << std::setw(12) << A32 << "  " << std::setw(12) << A33 << std::endl;
#endif
#ifdef USE_AVX_IMPLEMENTATION
_mm256_storeu_pd(buf, Va11);
A11 = buf[0];
_mm256_storeu_pd(buf, Va21);
A21 = buf[0];
_mm256_storeu_pd(buf, Va31);
A31 = buf[0];
_mm256_storeu_pd(buf, Va12);
A12 = buf[0];
_mm256_storeu_pd(buf, Va22);
A22 = buf[0];
_mm256_storeu_pd(buf, Va32);
A32 = buf[0];
_mm256_storeu_pd(buf, Va13);
A13 = buf[0];
_mm256_storeu_pd(buf, Va23);
A23 = buf[0];
_mm256_storeu_pd(buf, Va33);
A33 = buf[0];
std::cout << "Vector A (after multiplying with V) =" << std::endl;
std::cout << std::setw(12) << A11 << "  " << std::setw(12) << A12 << "  " << std::setw(12) << A13 << std::endl;
std::cout << std::setw(12) << A21 << "  " << std::setw(12) << A22 << "  " << std::setw(12) << A23 << std::endl;
std::cout << std::setw(12) << A31 << "  " << std::setw(12) << A32 << "  " << std::setw(12) << A33 << std::endl;
#endif
#endif

//###########################################################
// Re-normalize quaternion for matrix V
//###########################################################

#ifdef COMPUTE_V_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vqvs, Vqvs);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vqvvx, Vqvvx);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp1, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vqvvy, Vqvvy);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp1, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vqvvz, Vqvvz);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_add_pd(Vtmp1, Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_sqrt_pd(Vtmp2);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_div_pd(Vone, Vtmp1);)

#ifdef PERFORM_STRICT_QUATERNION_RENORMALIZATION
ENABLE_AVX_IMPLEMENTATION(Vtmp4 = _mm256_mul_pd(Vtmp1, Vone_half);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp1, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp1, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vtmp2, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_add_pd(Vtmp1, Vtmp4);)
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_sub_pd(Vtmp1, Vtmp3);)
#endif

ENABLE_AVX_IMPLEMENTATION(Vqvs = _mm256_mul_pd(Vqvs, Vtmp1);)
ENABLE_AVX_IMPLEMENTATION(Vqvvx = _mm256_mul_pd(Vqvvx, Vtmp1);)
ENABLE_AVX_IMPLEMENTATION(Vqvvy = _mm256_mul_pd(Vqvvy, Vtmp1);)
ENABLE_AVX_IMPLEMENTATION(Vqvvz = _mm256_mul_pd(Vqvvz, Vtmp1);)

#ifdef PRINT_DEBUGGING_OUTPUT
#ifdef USE_SCALAR_IMPLEMENTATION
std::cout << "Scalar qV =" << std::endl;
std::cout << std::setw(12) << Sqvs.f << "  " << std::setw(12) << Sqvvx.f << "  " << std::setw(12) << Sqvvy.f << "  " << std::setw(12) << Sqvvz.f << std::endl;
#endif
#ifdef USE_SSE_IMPLEMENTATION
_mm_storeu_ps(buf, Vqvs);
QVS = buf[0];
_mm_storeu_ps(buf, Vqvvx);
QVVX = buf[0];
_mm_storeu_ps(buf, Vqvvy);
QVVY = buf[0];
_mm_storeu_ps(buf, Vqvvz);
QVVZ = buf[0];
std::cout << "Vector qV =" << std::endl;
std::cout << std::setw(12) << QVS << "  " << std::setw(12) << QVVX << "  " << std::setw(12) << QVVY << "  " << std::setw(12) << QVVZ << std::endl;
#endif
#ifdef USE_AVX_IMPLEMENTATION
_mm256_storeu_pd(buf, Vqvs);
QVS = buf[0];
_mm256_storeu_pd(buf, Vqvvx);
QVVX = buf[0];
_mm256_storeu_pd(buf, Vqvvy);
QVVY = buf[0];
_mm256_storeu_pd(buf, Vqvvz);
QVVZ = buf[0];
std::cout << "Vector qV =" << std::endl;
std::cout << std::setw(12) << QVS << "  " << std::setw(12) << QVVX << "  " << std::setw(12) << QVVY << "  " << std::setw(12) << QVVZ << std::endl;
#endif
#endif
#endif

//###########################################################
// Construct QR factorization of A*V (=U*D) using Givens rotations
//###########################################################

#ifdef COMPUTE_U_AS_MATRIX
ENABLE_AVX_IMPLEMENTATION(Vu11 = Vone;)
ENABLE_AVX_IMPLEMENTATION(Vu21 = _mm256_xor_pd(Vu21, Vu21);)
ENABLE_AVX_IMPLEMENTATION(Vu31 = _mm256_xor_pd(Vu31, Vu31);)
ENABLE_AVX_IMPLEMENTATION(Vu12 = _mm256_xor_pd(Vu12, Vu12);)
ENABLE_AVX_IMPLEMENTATION(Vu22 = Vone;)
ENABLE_AVX_IMPLEMENTATION(Vu32 = _mm256_xor_pd(Vu32, Vu32);)
ENABLE_AVX_IMPLEMENTATION(Vu13 = _mm256_xor_pd(Vu13, Vu13);)
ENABLE_AVX_IMPLEMENTATION(Vu23 = _mm256_xor_pd(Vu23, Vu23);)
ENABLE_AVX_IMPLEMENTATION(Vu33 = Vone;)
#endif

#ifdef COMPUTE_U_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(Vqus = Vone;)
ENABLE_AVX_IMPLEMENTATION(Vquvx = _mm256_xor_pd(Vquvx, Vquvx);)
ENABLE_AVX_IMPLEMENTATION(Vquvy = _mm256_xor_pd(Vquvy, Vquvy);)
ENABLE_AVX_IMPLEMENTATION(Vquvz = _mm256_xor_pd(Vquvz, Vquvz);)
#endif

// First Givens rotation

#define SAPIVOT Sa11
#define SANPIVOT Sa21
#define SA11 Sa11
#define SA21 Sa21
#define SA12 Sa12
#define SA22 Sa22
#define SA13 Sa13
#define SA23 Sa23
#define SU11 Su11
#define SU12 Su12
#define SU21 Su21
#define SU22 Su22
#define SU31 Su31
#define SU32 Su32

#define VAPIVOT Va11
#define VANPIVOT Va21
#define VA11 Va11
#define VA21 Va21
#define VA12 Va12
#define VA22 Va22
#define VA13 Va13
#define VA23 Va23
#define VU11 Vu11
#define VU12 Vu12
#define VU21 Vu21
#define VU22 Vu22
#define VU31 Vu31
#define VU32 Vu32

#include "Singular_Value_Decomposition_Givens_QR_Factorization_Kernel.hpp"

#undef SAPIVOT
#undef SANPIVOT
#undef SA11
#undef SA21
#undef SA12
#undef SA22
#undef SA13
#undef SA23
#undef SU11
#undef SU12
#undef SU21
#undef SU22
#undef SU31
#undef SU32

#undef VAPIVOT
#undef VANPIVOT
#undef VA11
#undef VA21
#undef VA12
#undef VA22
#undef VA13
#undef VA23
#undef VU11
#undef VU12
#undef VU21
#undef VU22
#undef VU31
#undef VU32

// Update quaternion representation of U

#ifdef COMPUTE_U_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(Vqus = Vch;)
ENABLE_AVX_IMPLEMENTATION(Vquvx = _mm256_xor_pd(Vquvx, Vquvx);)
ENABLE_AVX_IMPLEMENTATION(Vquvy = _mm256_xor_pd(Vquvy, Vquvy);)
ENABLE_AVX_IMPLEMENTATION(Vquvz = Vsh;)
#endif

// Second Givens rotation

#define SAPIVOT Sa11
#define SANPIVOT Sa31
#define SA11 Sa11
#define SA21 Sa31
#define SA12 Sa12
#define SA22 Sa32
#define SA13 Sa13
#define SA23 Sa33
#define SU11 Su11
#define SU12 Su13
#define SU21 Su21
#define SU22 Su23
#define SU31 Su31
#define SU32 Su33

#define VAPIVOT Va11
#define VANPIVOT Va31
#define VA11 Va11
#define VA21 Va31
#define VA12 Va12
#define VA22 Va32
#define VA13 Va13
#define VA23 Va33
#define VU11 Vu11
#define VU12 Vu13
#define VU21 Vu21
#define VU22 Vu23
#define VU31 Vu31
#define VU32 Vu33

#include "Singular_Value_Decomposition_Givens_QR_Factorization_Kernel.hpp"

#undef SAPIVOT
#undef SANPIVOT
#undef SA11
#undef SA21
#undef SA12
#undef SA22
#undef SA13
#undef SA23
#undef SU11
#undef SU12
#undef SU21
#undef SU22
#undef SU31
#undef SU32

#undef VAPIVOT
#undef VANPIVOT
#undef VA11
#undef VA21
#undef VA12
#undef VA22
#undef VA13
#undef VA23
#undef VU11
#undef VU12
#undef VU21
#undef VU22
#undef VU31
#undef VU32

// Update quaternion representation of U

#ifdef COMPUTE_U_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(Vquvx = _mm256_mul_pd(Vsh, Vquvz);)
ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_mul_pd(Vsh, Vqus);)
ENABLE_AVX_IMPLEMENTATION(Vquvy = _mm256_sub_pd(Vquvy, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vqus = _mm256_mul_pd(Vch, Vqus);)
ENABLE_AVX_IMPLEMENTATION(Vquvz = _mm256_mul_pd(Vch, Vquvz);)
#endif

// Third Givens rotation

#define SAPIVOT Sa22
#define SANPIVOT Sa32
#define SA11 Sa21
#define SA21 Sa31
#define SA12 Sa22
#define SA22 Sa32
#define SA13 Sa23
#define SA23 Sa33
#define SU11 Su12
#define SU12 Su13
#define SU21 Su22
#define SU22 Su23
#define SU31 Su32
#define SU32 Su33

#define VAPIVOT Va22
#define VANPIVOT Va32
#define VA11 Va21
#define VA21 Va31
#define VA12 Va22
#define VA22 Va32
#define VA13 Va23
#define VA23 Va33
#define VU11 Vu12
#define VU12 Vu13
#define VU21 Vu22
#define VU22 Vu23
#define VU31 Vu32
#define VU32 Vu33

#include "Singular_Value_Decomposition_Givens_QR_Factorization_Kernel.hpp"

#undef SAPIVOT
#undef SANPIVOT
#undef SA11
#undef SA21
#undef SA12
#undef SA22
#undef SA13
#undef SA23
#undef SU11
#undef SU12
#undef SU21
#undef SU22
#undef SU31
#undef SU32

#undef VAPIVOT
#undef VANPIVOT
#undef VA11
#undef VA21
#undef VA12
#undef VA22
#undef VA13
#undef VA23
#undef VU11
#undef VU12
#undef VU21
#undef VU22
#undef VU31
#undef VU32

// Update quaternion representation of U

#ifdef COMPUTE_U_AS_QUATERNION
ENABLE_AVX_IMPLEMENTATION(Vtmp1 = _mm256_mul_pd(Vsh, Vquvx);)
ENABLE_AVX_IMPLEMENTATION(Vtmp2 = _mm256_mul_pd(Vsh, Vquvy);)
ENABLE_AVX_IMPLEMENTATION(Vtmp3 = _mm256_mul_pd(Vsh, Vquvz);)
ENABLE_AVX_IMPLEMENTATION(Vsh = _mm256_mul_pd(Vsh, Vqus);)
ENABLE_AVX_IMPLEMENTATION(Vqus = _mm256_mul_pd(Vch, Vqus);)
ENABLE_AVX_IMPLEMENTATION(Vquvx = _mm256_mul_pd(Vch, Vquvx);)
ENABLE_AVX_IMPLEMENTATION(Vquvy = _mm256_mul_pd(Vch, Vquvy);)
ENABLE_AVX_IMPLEMENTATION(Vquvz = _mm256_mul_pd(Vch, Vquvz);)
ENABLE_AVX_IMPLEMENTATION(Vquvx = _mm256_add_pd(Vquvx, Vsh);)
ENABLE_AVX_IMPLEMENTATION(Vqus = _mm256_sub_pd(Vqus, Vtmp1);)
ENABLE_AVX_IMPLEMENTATION(Vquvy = _mm256_add_pd(Vquvy, Vtmp3);)
ENABLE_AVX_IMPLEMENTATION(Vquvz = _mm256_sub_pd(Vquvz, Vtmp2);)
#endif

#ifdef COMPUTE_U_AS_MATRIX
#ifdef PRINT_DEBUGGING_OUTPUT
#ifdef USE_SCALAR_IMPLEMENTATION
std::cout << "Scalar U =" << std::endl;
std::cout << std::setw(12) << Su11.f << "  " << std::setw(12) << Su12.f << "  " << std::setw(12) << Su13.f << std::endl;
std::cout << std::setw(12) << Su21.f << "  " << std::setw(12) << Su22.f << "  " << std::setw(12) << Su23.f << std::endl;
std::cout << std::setw(12) << Su31.f << "  " << std::setw(12) << Su32.f << "  " << std::setw(12) << Su33.f << std::endl;
#endif
#ifdef USE_SSE_IMPLEMENTATION
_mm_storeu_ps(buf, Vu11);
U11 = buf[0];
_mm_storeu_ps(buf, Vu21);
U21 = buf[0];
_mm_storeu_ps(buf, Vu31);
U31 = buf[0];
_mm_storeu_ps(buf, Vu12);
U12 = buf[0];
_mm_storeu_ps(buf, Vu22);
U22 = buf[0];
_mm_storeu_ps(buf, Vu32);
U32 = buf[0];
_mm_storeu_ps(buf, Vu13);
U13 = buf[0];
_mm_storeu_ps(buf, Vu23);
U23 = buf[0];
_mm_storeu_ps(buf, Vu33);
U33 = buf[0];
std::cout << "Vector U =" << std::endl;
std::cout << std::setw(12) << U11 << "  " << std::setw(12) << U12 << "  " << std::setw(12) << U13 << std::endl;
std::cout << std::setw(12) << U21 << "  " << std::setw(12) << U22 << "  " << std::setw(12) << U23 << std::endl;
std::cout << std::setw(12) << U31 << "  " << std::setw(12) << U32 << "  " << std::setw(12) << U33 << std::endl;
#endif
#ifdef USE_AVX_IMPLEMENTATION
_mm256_storeu_pd(buf, Vu11);
U11 = buf[0];
_mm256_storeu_pd(buf, Vu21);
U21 = buf[0];
_mm256_storeu_pd(buf, Vu31);
U31 = buf[0];
_mm256_storeu_pd(buf, Vu12);
U12 = buf[0];
_mm256_storeu_pd(buf, Vu22);
U22 = buf[0];
_mm256_storeu_pd(buf, Vu32);
U32 = buf[0];
_mm256_storeu_pd(buf, Vu13);
U13 = buf[0];
_mm256_storeu_pd(buf, Vu23);
U23 = buf[0];
_mm256_storeu_pd(buf, Vu33);
U33 = buf[0];
std::cout << "Vector U =" << std::endl;
std::cout << std::setw(12) << U11 << "  " << std::setw(12) << U12 << "  " << std::setw(12) << U13 << std::endl;
std::cout << std::setw(12) << U21 << "  " << std::setw(12) << U22 << "  " << std::setw(12) << U23 << std::endl;
std::cout << std::setw(12) << U31 << "  " << std::setw(12) << U32 << "  " << std::setw(12) << U33 << std::endl;
#endif
#endif
#endif

#ifdef PRINT_DEBUGGING_OUTPUT
#ifdef USE_SCALAR_IMPLEMENTATION
std::cout << "Scalar A (after multiplying with U-transpose and V) =" << std::endl;
std::cout << std::setw(12) << Sa11.f << "  " << std::setw(12) << Sa12.f << "  " << std::setw(12) << Sa13.f << std::endl;
std::cout << std::setw(12) << Sa21.f << "  " << std::setw(12) << Sa22.f << "  " << std::setw(12) << Sa23.f << std::endl;
std::cout << std::setw(12) << Sa31.f << "  " << std::setw(12) << Sa32.f << "  " << std::setw(12) << Sa33.f << std::endl;
#endif
#ifdef USE_SSE_IMPLEMENTATION
_mm_storeu_ps(buf, Va11);
A11 = buf[0];
_mm_storeu_ps(buf, Va21);
A21 = buf[0];
_mm_storeu_ps(buf, Va31);
A31 = buf[0];
_mm_storeu_ps(buf, Va12);
A12 = buf[0];
_mm_storeu_ps(buf, Va22);
A22 = buf[0];
_mm_storeu_ps(buf, Va32);
A32 = buf[0];
_mm_storeu_ps(buf, Va13);
A13 = buf[0];
_mm_storeu_ps(buf, Va23);
A23 = buf[0];
_mm_storeu_ps(buf, Va33);
A33 = buf[0];
std::cout << "Vector A (after multiplying with U-transpose and V) =" << std::endl;
std::cout << std::setw(12) << A11 << "  " << std::setw(12) << A12 << "  " << std::setw(12) << A13 << std::endl;
std::cout << std::setw(12) << A21 << "  " << std::setw(12) << A22 << "  " << std::setw(12) << A23 << std::endl;
std::cout << std::setw(12) << A31 << "  " << std::setw(12) << A32 << "  " << std::setw(12) << A33 << std::endl;
#endif
#ifdef USE_AVX_IMPLEMENTATION
_mm256_storeu_pd(buf, Va11);
A11 = buf[0];
_mm256_storeu_pd(buf, Va21);
A21 = buf[0];
_mm256_storeu_pd(buf, Va31);
A31 = buf[0];
_mm256_storeu_pd(buf, Va12);
A12 = buf[0];
_mm256_storeu_pd(buf, Va22);
A22 = buf[0];
_mm256_storeu_pd(buf, Va32);
A32 = buf[0];
_mm256_storeu_pd(buf, Va13);
A13 = buf[0];
_mm256_storeu_pd(buf, Va23);
A23 = buf[0];
_mm256_storeu_pd(buf, Va33);
A33 = buf[0];
std::cout << "Vector A (after multiplying with U-transpose and V) =" << std::endl;
std::cout << std::setw(12) << A11 << "  " << std::setw(12) << A12 << "  " << std::setw(12) << A13 << std::endl;
std::cout << std::setw(12) << A21 << "  " << std::setw(12) << A22 << "  " << std::setw(12) << A23 << std::endl;
std::cout << std::setw(12) << A31 << "  " << std::setw(12) << A32 << "  " << std::setw(12) << A33 << std::endl;
#endif
#endif

#ifdef COMPUTE_U_AS_QUATERNION
#ifdef PRINT_DEBUGGING_OUTPUT
#ifdef USE_SCALAR_IMPLEMENTATION
std::cout << "Scalar qU =" << std::endl;
std::cout << std::setw(12) << Squs.f << "  " << std::setw(12) << Squvx.f << "  " << std::setw(12) << Squvy.f << "  " << std::setw(12) << Squvz.f << std::endl;
#endif
#ifdef USE_SSE_IMPLEMENTATION
_mm_storeu_ps(buf, Vqus);
QUS = buf[0];
_mm_storeu_ps(buf, Vquvx);
QUVX = buf[0];
_mm_storeu_ps(buf, Vquvy);
QUVY = buf[0];
_mm_storeu_ps(buf, Vquvz);
QUVZ = buf[0];
std::cout << "Vector qU =" << std::endl;
std::cout << std::setw(12) << QUS << "  " << std::setw(12) << QUVX << "  " << std::setw(12) << QUVY << "  " << std::setw(12) << QUVZ << std::endl;
#endif
#ifdef USE_AVX_IMPLEMENTATION
_mm256_storeu_pd(buf, Vqus);
QUS = buf[0];
_mm256_storeu_pd(buf, Vquvx);
QUVX = buf[0];
_mm256_storeu_pd(buf, Vquvy);
QUVY = buf[0];
_mm256_storeu_pd(buf, Vquvz);
QUVZ = buf[0];
std::cout << "Vector qU =" << std::endl;
std::cout << std::setw(12) << QUS << "  " << std::setw(12) << QUVX << "  " << std::setw(12) << QUVY << "  " << std::setw(12) << QUVZ << std::endl;
#endif
#endif
#endif

#ifdef __INTEL_COMPILER
#pragma warning(default : 592)
#endif
