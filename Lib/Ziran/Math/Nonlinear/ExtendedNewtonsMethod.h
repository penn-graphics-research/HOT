#ifndef EXTENDED_NEWTONS_METHOD_H
#define EXTENDED_NEWTONS_METHOD_H
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
class ExtendedNewtonsMethod {
    using Vec = typename Objective::NewtonVector;
    using T = typename Objective::Scalar;

    Objective& objective;

public:
    Vec step_direction;
    Vec residual;

    T tolerance;
    int max_iterations;

    ExtendedNewtonsMethod(Objective& objective, const T tolerance = (T)1e-6, const int max_iterations = 5)
        : objective(objective)
        , tolerance(tolerance)
        , max_iterations(max_iterations)
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
            ZIRAN_CONTEXT("Newton step", it);
            ZIRAN_INFO("Newton iter ", it);
            ZIRAN_INFO("This nonlinear iter is having residual l2 norm", residual_norm, " tolerance ", tolerance);
            if (objective.shouldExitByCN(residual)) {
                ZIRAN_INFO("Newton terminates at ", it);
                return true;
            }
            /// gast15
            T suggest_linear_solve_relative_tolerance = std::min((T)0.5, std::sqrt(std::max(residual_norm, (T)tolerance)));
            ZIRAN_INFO("Newton-suggested linear solve relative tolerance: ", suggest_linear_solve_relative_tolerance, " and tolerance is ", tolerance);
            objective.computeStep(step_direction, residual, suggest_linear_solve_relative_tolerance);
            objective.recoverSolution(step_direction);
            x += step_direction;
            objective.transformResidual(step_direction);
        }
        return false;
    }
}; // namespace ZIRAN
} // namespace ZIRAN
#endif
