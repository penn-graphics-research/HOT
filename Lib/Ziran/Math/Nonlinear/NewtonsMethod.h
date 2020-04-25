#ifndef NEWTONS_METHOD_H
#define NEWTONS_METHOD_H
#include <Ziran/CS/Util/ErrorContext.h>
#include <Ziran/CS/Util/Meta.h>
#include <Ziran/CS/Util/Timer.h>
#include <iostream>

namespace ZIRAN {
/**
  Newton's Method
  Templatized on TOBJ the objective function
  which should implement the following functions
  T computeGradient(TV& x)
  TV& computeStep()  // computes the next step based on the previous x passed into computeResidual
  and typedefs
  TV and T
*/
template <class Objective>
class NewtonsMethod {
    using Vec = typename Objective::NewtonVector;
    using T = typename Objective::Scalar;

    Objective& objective;

public:
    Vec step_direction;
    Vec residual;

    T tolerance;
    int max_iterations;
    T linear_solve_tolerance_scale;

    NewtonsMethod(Objective& objective, const T tolerance = (T)1e-6, const int max_iterations = 5)
        : objective(objective)
        , tolerance(tolerance)
        , max_iterations(max_iterations)
        , linear_solve_tolerance_scale((T)1)
    {
    }

    bool solve(Vec& x, const bool verbose = false)
    {
        ZIRAN_TIMER();
        step_direction.resizeLike(x);
        residual.resizeLike(x);
        //objective.evaluatePerNodeCNTolerance(tolerance);
        for (int it = 0; it < max_iterations; it++) {
            objective.updateState(x);
            objective.computeResidual(residual);
            T residual_norm = objective.computeNorm(residual);
            ZIRAN_INFO("\n\n");
            ZIRAN_INFO("Newton iter ", it);
            // ZIRAN_INFO("Newton iter ", it, "; (designated norm) residual = ", residual_norm);
            // ZIRAN_VERB_IF(verbose, "Newton iter ", it, "; (designated norm) residual = ", residual_norm);
            ZIRAN_CONTEXT("Newton step", it);
            ZIRAN_INFO("This nonlinear iter is having residual l2 norm", residual_norm, " tolerance ", tolerance);
            // ZIRAN_CONTEXT(residual_norm);
            if (residual_norm < tolerance) {
                ZIRAN_INFO("Newton terminates at ", it);
                // ZIRAN_VERB_IF(true, "Newton terminates at ", it, "; (designated norm) residual = ", residual_norm);
                return true;
            }
            // else {
            //     ZIRAN_VERB_IF(true, "Newton residual = ", residual_norm);
            // }
#if 0
            /// original HOT intended metric
            T linear_solve_relative_tolerance = std::min((T)0.5, linear_solve_tolerance_scale * std::sqrt(std::max(residual_norm / tolerance, (T)1)));
#elif 1
            /// gast15
            T linear_solve_relative_tolerance = std::min((T)0.5, std::sqrt(std::max(residual_norm, (T)tolerance)));
#endif
            ZIRAN_INFO("linear solve relative tolerance: ", linear_solve_relative_tolerance, " and CN tolerance is ", tolerance);
            objective.computeStep(step_direction, residual, linear_solve_relative_tolerance);
            objective.recoverSolution(step_direction);
            x += step_direction;
            objective.transformResidual(step_direction);
        }
        return false;
    }
}; // namespace ZIRAN
} // namespace ZIRAN
#endif
