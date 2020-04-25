#ifndef INEXACT_CONJUGATE_GRADIENT_H
#define INEXACT_CONJUGATE_GRADIENT_H

#include <Ziran/Math/Linear/KrylovSolvers.h>
#include <tbb/tbb.h>

#include "LinearSolver.h"

namespace ZIRAN {
template <class T, class TM, class TV>

//
// TODO: This CG seems to be buggy! Fix it using PhysBAM reference.
//
class InexactConjugateGradient : public LinearSolver<T, TM, TV> {

    using Base = LinearSolver<T, TM, TV>;
    using Vec = Vector<T, Eigen::Dynamic>;

    /** All notations adopted from Wikipedia, 
     * q denotes A*p in general */
    TV r, p, q, temp;

public:
    bool useCharacteristicNorm;
    Vec mass;
    TV mq;

    T gast15tau;

    InexactConjugateGradient(const int max_it_input)
        : Base(max_it_input)
    {
        useCharacteristicNorm = false;
        gast15tau = 0;
    }

    ~InexactConjugateGradient() {}

    void reinitialize(const TV& b)
    {
        r.resizeLike(b);
        p.resizeLike(b);
        q.resizeLike(b);
        mq.resizeLike(b);
        temp.resizeLike(b);
    }

    int solve(const TM& A, TV& x, const TV& b, const bool verbose = false)
    {
        //TODO: adaptive tolerance on unpreconditioned residual norm

        ZIRAN_QUIET_TIMER();
        assert(x.size() == b.size());
        reinitialize(b);
        int cnt = 0;
        T alpha, beta, residual_preconditioned_norm, zTrk, zTrk_last;
        T preconditionedNorm = 0;

        //NOTE: requires that the input x has been projected
        A.multiply(x, temp);
        r = b - temp;
        A.project(r);
        A.precondition(r, q); //NOTE: requires that preconditioning matrix is projected
        p = q;

        zTrk = Base::dotProduct(r, q);
        residual_preconditioned_norm = std::sqrt(zTrk);

        // Inexact Newton combining Algorithm 7.1 of Nocedal and Wright book and Gast 15
        T forcing_sequence = std::min((T)0.5, std::sqrt(std::max(residual_preconditioned_norm, Base::tolerance)));
        T local_tolerance = forcing_sequence * residual_preconditioned_norm;
        ZIRAN_INFO("Inexact CG forcing sequence: residual_preconditioned_norm = ", residual_preconditioned_norm, ", cgtolerance = ", Base::tolerance, ", forcing sequence = ", forcing_sequence);

        for (cnt = 0; cnt < Base::max_iterations; ++cnt) {
            if (residual_preconditioned_norm < local_tolerance) {
                ZIRAN_VERB_IF(verbose, "\tInexact CG terminates at ", cnt, "; (preconditioned norm) residual = ", residual_preconditioned_norm, " local tolerance: ", local_tolerance, " with forcing sequence ", forcing_sequence);
                return cnt;
            }

            if (cnt % 50 == 0) {
                ZIRAN_VERB_IF(verbose, "\tInexact CG iter ", cnt, "; (preconditioned norm) residual = ", residual_preconditioned_norm, " local tolerance: ", local_tolerance, " with forcing sequence ", forcing_sequence);
            }

            A.multiply(p, temp);
            A.project(temp);
            alpha = zTrk / Base::dotProduct(temp, p);

            x += p * alpha;
            r -= temp * alpha;
            A.precondition(r, q); //NOTE: requires that preconditioning matrix is projected

            zTrk_last = zTrk;
            zTrk = Base::dotProduct(q, r);
            beta = zTrk / zTrk_last;

            p = q + beta * p;

            residual_preconditioned_norm = std::sqrt(zTrk);
        }
        ZIRAN_VERB_IF(verbose, "Inexact ConjugateGradient max iterations reached ", Base::max_iterations);
        return Base::max_iterations;
    }
};
} // namespace ZIRAN

#endif
