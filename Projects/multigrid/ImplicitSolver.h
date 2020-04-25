#ifndef IMPLICIT_SOLVER_H
#define IMPLICIT_SOLVER_H

#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/KrylovSolvers.h>
#include <Ziran/Math/Linear/Minres.h>
#include <Ziran/Math/Linear/ConjugateGradient.h>
#include <Ziran/Math/Linear/InexactConjugateGradient.h>
#include <Ziran/Math/Linear/GeneralizedMinimalResidual.h>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Sim/SimulationBase.h>
#include <Ziran/Math/Linear/EigenSparseLU.h>

#include <set>
#include <tbb/tbb.h>
#include "Configurations.h"

#include "SparseMatrixFast.h"

namespace ZIRAN {

/**
   Simulation owns Objective and NewtonsMethod.
   This will allow options to do Newton.
   It stores the parameters for them. But it is a little weird.
   This is supposed to be used by both MPM and Elasticity.
 */
template <class Simulation>
class ImplicitSolverObjective {
public:
    using T = typename Simulation::Scalar;
    static const int dim = Simulation::dim;
    using Objective = ImplicitSolverObjective<Simulation>;
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using Hessian = Eigen::Matrix<T, dim * dim, dim * dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using Vec = Vector<T, Eigen::Dynamic>;

    using Scalar = T;
    using NewtonVector = TVStack;

    Simulation& simulation;

    std::function<void(TVStack&)> project;
    std::function<void(const TVStack&, TVStack&)> precondition;

    GeneralizedMinimalResidual<T, Objective, TVStack> gmres;
    Minres<T, Objective, TVStack> minres;
#if USE_GAST15_METRIC
    ConjugateGradient<T, Objective, TVStack> cg;
#else
    InexactConjugateGradient<T, Objective, TVStack> cg;
#endif
    bool matrix_free;
    bool lu;
    bool isNewStep{ false };

    TVStack dv0, dRhs, rhs; ///< initial dv at each Newton iteration
    T Ek;
    bool updated;
    std::vector<int> entryCol;
    std::vector<TM> entryVal;
    inline static std::vector<TM> diagVal;
    std::vector<IV> id2coord;
    //std::vector<T> nodeCNTol;
    Vec nodeCNTol;
    std::vector<T> particleInitialCN;
    SparseMatrix<T, dim> matrix;
    int curIter;
    Timer timer;

    ImplicitSolverObjective(Simulation& simulation)
        : simulation(simulation)
        , project([](TVStack&) {})
        , precondition([](const TVStack& x, TVStack& b) { b = x; })
        , gmres(20)
        , minres(20)
        , cg(10000)
        , matrix_free(false)
        , lu(false)
        , matrix(entryCol, entryVal, id2coord, precondition)
    {
        updated = false;
        gmres.setTolerance(1);
        minres.setTolerance(1);
        cg.setTolerance(1);
    }

    ImplicitSolverObjective(const ImplicitSolverObjective& objective) = delete;

    ~ImplicitSolverObjective() {}

    template <class Projection>
    void initialize(const Projection& projection_function)
    {
        ZIRAN_INFO("Krylov matrix free : ", matrix_free);
        project = projection_function;
    }

    void reinitialize()
    {
    }

    void recoverSolution(TVStack& ddv)
    {
        if (HOTSettings::systemBCProject && HOTSettings::boundaryType == 1)
            for (auto iter = simulation.collision_nodes.begin(); iter != simulation.collision_nodes.end(); ++iter) {
                int node_id = iter->node_id;
                if (iter->shouldRotate) {
                    ddv.col(node_id) = iter->Rinv * ddv.col(node_id);
                }
            }
    }

    void transformResidual(TVStack& r)
    {
        if (HOTSettings::systemBCProject && HOTSettings::boundaryType == 1)
            for (auto iter = simulation.collision_nodes.begin(); iter != simulation.collision_nodes.end(); ++iter) {
                int node_id = iter->node_id;
                if (iter->shouldRotate)
                    r.col(node_id) = iter->R * r.col(node_id);
            }
    }

    // called by Newton
    void computeResidual(TVStack& residual, bool force = false)
    {
        ZIRAN_TIMER();
        ZIRAN_ASSERT(residual.cols() == simulation.num_nodes);
        if (updated && !force) return;
        tbb::parallel_for(tbb::blocked_range<int>(0, residual.cols(), 256),
            [&](const tbb::blocked_range<int>& range) {
                int start = range.begin();
                int length = (range.end() - range.begin());
                TV dtg = simulation.dt * simulation.gravity;
                residual.middleCols(start, length) = dtg * simulation.mass_matrix.segment(start, length).transpose();
            });

        simulation.addScaledForces(simulation.dt, residual);

        if (!simulation.quasistatic) {
            simulation.inertia->addScaledForces(simulation.dt, residual);
        }

        transformResidual(residual);
        project(residual);
        rhs = residual;

        if (simulation.full_implicit) {
            simulation.force->restoreStrain();
            simulation.force->evolveStrainWithDt();
        }
    }

    // called by Newton
    T computeNorm(const TVStack& residual) const
    {
        ZIRAN_ASSERT(residual.cols() == simulation.num_nodes);
        {
            T norm_sq = tbb::parallel_reduce(tbb::blocked_range<int>(0, residual.cols(), 256), (T)0, [&](const tbb::blocked_range<int>& range, T ns) -> T {
                T rn = 0;
                for (int i = range.begin(); i < range.end() && i < residual.cols(); ++i) {
                    rn += residual.col(i).squaredNorm();
                }
                return ns + rn; },
                [](T a, T b) -> T { return a + b; });
            return std::sqrt(norm_sq);
        }
    }

    // called by Newton or LBFGS
    bool shouldExitByCN(const TVStack& residual)
    {
        {
            T timeAccum;
            if (curIter++ == 0) {
                timer.start();
                timeAccum = 0;
            }
            else {
                timeAccum = timer.click(false).count();
            }
            if (!HOTSettings::useCN) {
                auto res = computeNorm(residual);
                ZIRAN_INFO("This nonlinear iter of Substep ", simulation.getSubstep(), " of Frame ", simulation.getFrame(), " is having residual l2 norm ", res);
                return res < HOTSettings::cneps;
            }
            ///scaled version
            ZIRAN_ASSERT(residual.cols() == simulation.num_nodes);
            T scaledNorm = tbb::parallel_reduce(tbb::blocked_range<int>(0, residual.cols(), 256), (T)0, [&](const tbb::blocked_range<int>& range, T ns) -> T {
                T rn = 0;
                for (int i = range.begin(); i < range.end() && i < residual.cols(); ++i)
                    rn += residual.col(i).squaredNorm() / (nodeCNTol[i] * nodeCNTol[i]);
                return ns + rn; },
                [](T a, T b) -> T { return a + b; });
            if (simulation.num_nodes == 0) {
                ZIRAN_INFO("Substep ", simulation.getSubstep(), " of Frame ", simulation.getFrame(), " exits with scaled residual l2 norm ", 0);
                return true;
            }
            bool shouldExit = scaledNorm < simulation.num_nodes;
            if (shouldExit)
                ZIRAN_INFO("This nonlinear iter of Substep ", simulation.getSubstep(), " of Frame ", simulation.getFrame(), " exits with scaled residual l2 norm ", std::sqrt(scaledNorm / simulation.num_nodes));
            else
                ZIRAN_INFO("This nonlinear iter of Substep ", simulation.getSubstep(), " of Frame ", simulation.getFrame(), " is having scaled residual l2 norm ", std::sqrt(scaledNorm / simulation.num_nodes));
            ZIRAN_INFO("iter ", curIter - 1, " @#$ of Substep ", simulation.getSubstep(), " of Frame ", simulation.getFrame(), " is at ", timeAccum, "s with scaled residual l2 norm ", std::sqrt(scaledNorm / simulation.num_nodes));

            return shouldExit;
        }
    }

    T innerProduct(const TVStack& ddv, const TVStack& residual) const
    {
        ZIRAN_ASSERT(residual.cols() == ddv.cols());
        T result = tbb::parallel_reduce(tbb::blocked_range<int>(0, residual.cols(), 256), (T)0,
            [&](const tbb::blocked_range<int>& range, T ns) -> T {
                int start = range.begin();
                int length = (range.end() - range.begin());
                const auto& ddv_block = ddv.middleCols(start, length);
                const auto& r_block = residual.middleCols(start, length);
                const auto& mass_block = simulation.mass_matrix.segment(start, length);
                // ZIRAN_DEBUG(ddv_block);
                // ZIRAN_DEBUG(r_block);
                // ZIRAN_DEBUG(mass_block);

                //T to_add = ((ddv_block.transpose() * r_block).diagonal().array() / mass_block.array()).sum();
                T to_add = ((ddv_block.transpose() * r_block).diagonal().array()).sum();
                // ZIRAN_DEBUG(to_add);
                return ns += to_add;
            },
            [](T a, T b) -> T { return a + b; });
        return result;
    }

    // called by Newton
    void updateState(const TVStack& dv, bool force = false)
    {
        ZIRAN_TIMER();
        ZIRAN_ASSERT(dv.cols() == simulation.num_nodes);
        if (updated && !force) return;
        simulation.moveNodes(dv);
        for (auto& lf : simulation.forces) {
            lf->updatePositionBasedState();
        }
        if (!simulation.quasistatic) {
            simulation.inertia->updatePositionBasedState();
        }

        if (HOTSettings::linesearch)
            Ek = totalEnergy();
    }

    T totalEnergy()
    {
        ZIRAN_TIMER();
        T result = 0;
        for (auto& lf : simulation.forces) {
            result += lf->totalEnergy();
        }
        if (!simulation.quasistatic) {
            result += simulation.inertia->totalEnergy();
        }

        result += tbb::parallel_reduce(tbb::blocked_range<int>(0, simulation.dv.cols(), 256), (T)0,
            [&](const tbb::blocked_range<int>& range, T ns) -> T {
                int start = range.begin();
                int length = (range.end() - range.begin());
                const auto& dv_block = simulation.dv.middleCols(start, length);
                const auto& mass_block = simulation.mass_matrix.segment(start, length);
                return ns - simulation.dt * ((simulation.gravity.transpose() * dv_block).dot(mass_block));
            },
            [](T a, T b) -> T { return a + b; });
        return result;
    }

    void resetLSFlag(TVStack& dv)
    {
        updated = false;
        dv0 = dv;
        curIter = 0;
    }

    void BCprojectionSanityCheck(const TVStack& sol, const TVStack& residual)
    {
        for (auto iter = simulation.collision_nodes.begin(); iter != simulation.collision_nodes.end(); ++iter) {
            int idx = iter->node_id;
            auto diff = (sol.col(idx) - residual.col(idx));
            if (sol.col(idx).norm() > 1e-10)
                ZIRAN_WARN("sol[", idx, "]: ", sol(0, idx), ", ", sol(1, idx), ", ", sol(2, idx));
            if (residual.col(idx).norm() > 1e-10)
                ZIRAN_WARN("residual[", idx, "]: ", residual(0, idx), ", ", residual(1, idx), ", ", residual(2, idx));
            if (diff.norm() > 1e-10)
                ZIRAN_WARN("res-sol[", idx, "]: ", diff(0, idx), ", ", diff(1, idx), ", ", diff(2, idx));
        }
    }

    T angle(TVStack& a, TVStack& b)
    {
        T na = a.norm();
        T nb = b.norm();
        T mul = innerProduct(a, b);
        ZIRAN_INFO("cosine of stepdirection & rhs: ", mul / na / nb);
        return mul / na / nb;
    }

    void checkMultigridSystemMatrix()
    {
        matrix.rebuildPreconditioner(simulation.mass_matrix, project, dRhs, HOTSettings::levelCnt, HOTSettings::times, HOTSettings::omega, nodeCNTol, true);
    }

    T lineSearch(TVStack& ddv, TVStack& residual, T alpha)
    {
        ZIRAN_TIMER();
        TVStack dvnew;
        dvnew.resizeLike(ddv);

        recoverSolution(ddv);
        T Ek0 = Ek;
        do {
            dvnew = dv0 + ddv * alpha;
            updateState(dvnew, true);
            alpha *= 0.5;
            ZIRAN_INFO("line search omega: ", alpha * 2, " testing Ek ", Ek, " with  ", Ek0);
        } while (Ek > Ek0);
        alpha *= 2;
        ddv *= alpha;
        transformResidual(ddv);
        computeResidual(residual, true);
        updated = true;
        dv0 = dvnew;
        return alpha;
    }

    void HinvApproxInit()
    {
        ZIRAN_TIMER();
        buildMatrix<true>();
        ZIRAN_ASSERT(!(!HOTSettings::systemBCProject && HOTSettings::levelCnt > 1));

        if (!HOTSettings::useBaselineMultigrid) {
            HOTSettings::cgratio = cg.tolerance;
            matrix.rebuildPreconditioner(simulation.mass_matrix, project, dRhs, HOTSettings::levelCnt, HOTSettings::times, HOTSettings::omega, nodeCNTol);
        }
        else {
            auto& mgp = matrix.rebuildPreconditioner(simulation.mass_matrix, project, dRhs, HOTSettings::levelCnt, HOTSettings::times, HOTSettings::omega, nodeCNTol, true);
            simulation.particlesToMultigrids(mgp);
            precondition = tl::function_ref<void(const TVStack&, TVStack&)>(mgp);
            //ZIRAN_WARN("Just built multigrids!");
            //getchar();
        }
        isNewStep = false;
    }

    void computeStep(TVStack& ddv, const TVStack& residual, const T linear_solve_relative_tolerance)
    {
        ZIRAN_ASSERT(ddv.cols() == simulation.num_nodes);
        ddv.setZero();
        if (!matrix_free) {
            /// bool systemBCProject
            if (HOTSettings::systemBCProject)
                buildMatrix<true>();
            else
                buildMatrix();
            if (!((HOTSettings::lsolver == 1 || HOTSettings::lsolver == 2) && HOTSettings::Ainv == 2)) {
                auto& mgp = matrix.rebuildPreconditioner(simulation.mass_matrix, project, dRhs, HOTSettings::levelCnt, HOTSettings::times, HOTSettings::omega, nodeCNTol, true);
                if (!HOTSettings::useBaselineMultigrid)
                    matrix.rebuildPreconditioner(simulation.mass_matrix, project, dRhs, HOTSettings::levelCnt, HOTSettings::times, HOTSettings::omega, nodeCNTol);
                else {
                    simulation.particlesToMultigrids(mgp);
                    precondition = tl::function_ref<void(const TVStack&, TVStack&)>(mgp);
                    //ZIRAN_WARN("Just built multigrids!");
                    //getchar();
                }
                /// force diagonal entry preconditioner
                if (HOTSettings::levelCnt == 1 && HOTSettings::times == 1) {
                    if (HOTSettings::Ainv == 0)
                        setPreconditioner([&](const TVStack& in, TVStack& out) {
                            tbb::parallel_for(int(0), (int)in.cols(), [&](int i) {
                                out.col(i) = mgp.sysmats[0]->_mat->diagonalEntry[i] * in.col(i);
                            });
                        });
                    else if (HOTSettings::Ainv == 1)
                        setPreconditioner([&](const TVStack& in, TVStack& out) {
                            tbb::parallel_for(int(0), (int)in.cols(), [&](int i) {
                                out.col(i) = mgp.sysmats[0]->_mat->diagonalBlock[i] * in.col(i);
                            });
                        });
                }
            }
        }
        else {
            buildDiagonal();
            setPreconditioner([&](const TVStack& in, TVStack& out) {
                tbb::parallel_for(int(0), (int)in.cols(), [&](int i) {
                    out.col(i) = diagVal[i] * in.col(i);
                });
            });
        }

        if (simulation.full_implicit) {
            gmres.setRelativeTolerance(linear_solve_relative_tolerance);
            gmres.solve(*this, ddv, residual, simulation.verbose);
        }
        else {
            if (HOTSettings::lsolver == 1) {
                //minres.mass = simulation.mass_matrix;
                minres.setRelativeTolerance(linear_solve_relative_tolerance);
                if (HOTSettings::systemBCProject)
                    const_cast<TVStack&>(residual) += dRhs;
                minres.solve(*this, ddv, residual, simulation.verbose);
            }
            else if (HOTSettings::lsolver == 2) {
                cg.mass = simulation.mass_matrix;
                cg.useCharacteristicNorm = true;
                cg.setRelativeTolerance(linear_solve_relative_tolerance);
                if (HOTSettings::systemBCProject)
                    const_cast<TVStack&>(residual) += dRhs;
                cg.solve(*this, ddv, residual, simulation.verbose);
            }
            else if (HOTSettings::lsolver == 4) {
                ZIRAN_ASSERT(HOTSettings::systemBCProject, "Must BC project system matrix to enable direct solver!");
                auto& mgp = matrix.rebuildPreconditioner(simulation.mass_matrix, project, dRhs, HOTSettings::levelCnt, HOTSettings::times, HOTSettings::omega, nodeCNTol, true);
                auto& sysmat = *mgp.sysmats[0]->_mat;
                sysmat.solve(residual, ddv);
            }
            if (HOTSettings::linesearch) {
                lineSearch(ddv, const_cast<TVStack&>(residual), 1.0);
            }
        }
        isNewStep = false;
    }

    T computeDescentStep(TVStack& ddv, const TVStack& residual, const T linear_solve_relative_tolerance)
    {
        computeStep(ddv, residual, linear_solve_relative_tolerance);

        T inner_prod = innerProduct(ddv, residual);
        T tol = 1e-5 * computeNorm(ddv) * computeNorm(residual);
        if (inner_prod < -tol) {
            ZIRAN_INFO("Newton step direction is of same direction as gradient. Look backwards!!");
            inner_prod = -inner_prod;
            ddv = -ddv;
        }
        else if (std::abs(inner_prod) < tol) {
            // ZIRAN_DEBUG(simulation.mass_matrix);
            // ZIRAN_DEBUG(ddv);
            // ZIRAN_DEBUG(residual);
            // ZIRAN_DEBUG(inner_prod);

            ZIRAN_INFO("Newton step direciton is almost orthogonal to gradient. Look towards negative gradient direction!");
            T res_norm = computeNorm(residual);
            T scale = computeNorm(ddv) / res_norm;
            ddv = scale * residual;
            inner_prod = scale * res_norm * res_norm;
        }
        else if (!std::isfinite(inner_prod)) {
            ddv = residual;
            T res_norm = computeNorm(residual);
            inner_prod = res_norm * res_norm;
        }
        return inner_prod;
    }

    static inline int linearOffset(const IV& node_offset)
    {
        return (node_offset[0] + 2) * 25 + (node_offset[1] + 2) * 5 + node_offset[2] + 2;
    }

    template <bool projectSystem = false>
    void buildMatrix()
    {
        ZIRAN_TIMER();
        id2coord.resize(simulation.num_nodes);
        simulation.grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            id2coord[g.idx] = node;
        });

        entryCol.resize(simulation.num_nodes * 125);
        entryVal.resize(simulation.num_nodes * 125);

        if constexpr (projectSystem) {
            dRhs.resizeLike(rhs);
            dRhs.setZero();
        }

        // inertia term
        T inertia_scale = T(1);
        simulation.grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            for (int i = 0; i < 125; ++i) {
                entryCol[g.idx * 125 + i] = -1;
                entryVal[g.idx * 125 + i] = TM::Zero();
            }
            entryCol[g.idx * 125 + linearOffset(IV::Zero())] = g.idx;
            entryVal[g.idx * 125 + linearOffset(IV::Zero())] = inertia_scale * simulation.mass_matrix(g.idx) * TM::Identity();
        });

        // force term
        T force_scale = simulation.dt * simulation.dt;
        auto* vol_pointer = &simulation.particles.DataManager::get(AttributeName<T>("element measure"));

        for (auto& h : simulation.force->helpers)
            h->runLambdaWithDifferential(simulation.particle_order, simulation.particle_group, simulation.block_offset,
                [&](int i, const Hessian& ddF, const TM& Fn, const T& ddJ, const T& Jn, bool use_J) {
                    auto& Xp = simulation.particles.X[i];
                    auto& vol = (*vol_pointer)[i];

                    BSplineWeights<T, dim> spline(Xp, simulation.dx);
                    std::vector<TV> cached_w(27);
                    std::vector<IV> cached_node(27);
                    std::vector<int> cached_idx(27);
                    int cnt = 0;
                    simulation.grid.iterateKernel(spline, simulation.particle_base_offset[i], [&](const IV& node, const T& w, const TV& dw, GridState<T, dim>& g) {
                        if (g.idx < 0)
                            return;
                        cached_w[cnt] = use_J ? dw : Fn.transpose() * dw;
                        cached_node[cnt] = node;
                        cached_idx[cnt++] = g.idx;
                    });

                    for (int i = 0; i < cnt; ++i) {
                        const TV& wi = cached_w[i];
                        const IV& nodei = cached_node[i];
                        const int& dofi = cached_idx[i];
                        ///
                        for (int j = 0; j < cnt; ++j) {
                            const TV& wj = cached_w[j];
                            const IV& nodej = cached_node[j];
                            const int& dofj = cached_idx[j];
                            if (dofj < dofi) continue;
                            TM dFdX = TM::Zero();
                            if (use_J)
                                dFdX += ddJ * Jn * Jn * wi * wj.transpose();
                            else
                                for (int q = 0; q < dim; q++)
                                    for (int v = 0; v < dim; v++) {
                                        dFdX += ddF.template block<dim, dim>(dim * v, dim * q) * wi(v) * wj(q);
                                    }
                            TM delta = force_scale * vol * dFdX;
                            ///

                            entryCol[dofi * 125 + linearOffset(nodei - nodej)] = dofj;
                            entryVal[dofi * 125 + linearOffset(nodei - nodej)] += delta;
                            if (dofi != dofj) {
                                entryCol[dofj * 125 + linearOffset(nodej - nodei)] = dofi;
                                entryVal[dofj * 125 + linearOffset(nodej - nodei)] += delta.transpose();
                            }
                        }
                    }
                },
                HOTSettings::useBaselineMultigrid ? 1 : 0);
        //matrixSanityCheck();

        if constexpr (projectSystem) {
            std::map<int, const CollisionNode<T, dim>&> bCollide;
            for (auto iter = simulation.collision_nodes.begin(); iter != simulation.collision_nodes.end(); ++iter) {
                bCollide.emplace(iter->node_id, *iter);
                //bCollide[iter->node_id] = *iter;
            }
            tbb::parallel_for(0, simulation.num_nodes, [&](int i) {
                int st = i * 125;
                int ed = st + 125;
                const auto& iIter = bCollide.find(i);
                bool iCollide = iIter != bCollide.end();
                bool iSlip = iCollide ? iIter->second.shouldRotate : false;
                for (; st < ed; ++st) {
                    int j = entryCol[st];
                    if (j == -1) { ///< zero-entry
                        entryCol[st] = i > 0 ? 0 : 1;
                        continue;
                    }
                    const auto& jIter = bCollide.find(j);
                    bool jCollide = jIter != bCollide.end();
                    if (!iCollide && !jCollide) ///< no projection
                        continue;
                    bool jSlip = jCollide ? jIter->second.shouldRotate : false;
                    auto& val = entryVal[st];
                    if ((iCollide && !iSlip) || (jCollide && !jSlip)) { ///< sticky projection
                        //dRhs.col(j) -= val * rhs.col(j);
                        if (j == i)
                            val = TM::Identity();
                        else
                            val = TM::Zero();
                        continue;
                    }
                    if (iSlip) val = iIter->second.R * val;
                    if (jSlip) val = val * jIter->second.Rinv;
                    if (iSlip) val(0, 0) = 0, val(0, 1) = 0, val(0, 2) = 0;
                    if (jSlip) val(0, 0) = 0, val(1, 0) = 0, val(2, 0) = 0;
                    if (i == j) val(0, 0) = 1;
                }
            });
        }
        else {
            tbb::parallel_for(0, simulation.num_nodes, [&](int i) {
                int st = i * 125;
                int ed = st + 125;
                for (; st < ed; ++st)
                    if (entryCol[st] == -1)
                        entryCol[st] = i > 0 ? 0 : 1;
            });
        }
    }

    void buildDiagonal()
    {
        ZIRAN_TIMER();

        entryVal.resize(simulation.num_nodes);
        // inertia term
        T inertia_scale = T(1);
        simulation.grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            entryVal[g.idx] = inertia_scale * simulation.mass_matrix(g.idx) * TM::Identity();
        });

        // force term
        T force_scale = simulation.dt * simulation.dt;
        auto* vol_pointer = &simulation.particles.DataManager::get(AttributeName<T>("element measure"));

        for (auto& h : simulation.force->helpers)
            h->runLambdaWithDifferential(simulation.particle_order, simulation.particle_group, simulation.block_offset,
                [&](int i, const Hessian& ddF, const TM& Fn, const T& ddJ, const T& Jn, bool use_J) {
                    auto& Xp = simulation.particles.X[i];
                    auto& vol = (*vol_pointer)[i];

                    BSplineWeights<T, dim> spline(Xp, simulation.dx);
                    std::vector<TV> cached_w(27);
                    std::vector<int> cached_idx(27);
                    int cnt = 0;
                    simulation.grid.iterateKernel(spline, simulation.particle_base_offset[i], [&](const IV& node, const T& w, const TV& dw, GridState<T, dim>& g) {
                        if (g.idx < 0)
                            return;
                        cached_w[cnt] = use_J ? dw : Fn.transpose() * dw;
                        cached_idx[cnt++] = g.idx;
                    });

                    for (int i = 0; i < cnt; ++i) {
                        const TV& wi = cached_w[i];
                        const int& dof = cached_idx[i];
                        ///
                        const TV& wj = cached_w[i];
                        TM dFdX = TM::Zero();
                        if (use_J)
                            dFdX += ddJ * Jn * Jn * wi * wj.transpose();
                        else
                            for (int q = 0; q < dim; q++)
                                for (int v = 0; v < dim; v++) {
                                    dFdX += ddF.template block<dim, dim>(dim * v, dim * q) * wi(v) * wj(q);
                                }
                        TM delta = force_scale * vol * dFdX;
                        ///
                        entryVal[dof] += delta;
                    }
                });
        //matrixSanityCheck();
        diagVal.resize(entryVal.size());
        if (HOTSettings::Ainv == 0)
            tbb::parallel_for(0, (int)entryVal.size(), [&](int i) {
                diagVal[i] = entryVal[i].diagonal().asDiagonal().inverse();
            });
        else
            tbb::parallel_for(0, (int)entryVal.size(), [&](int i) {
                diagVal[i] = entryVal[i].inverse();
            });
    }

    void evaluatePerNodeCNTolerance(T eps, T dt)
    {
        ZIRAN_TIMER();
        //T dt = simulation.dt;
        nodeCNTol.resize(simulation.num_nodes);
        nodeCNTol.setZero();
        //std::fill(nodeCNTol.begin(), nodeCNTol.end(), (T)0);
        //particleInitialCN.resize();

        auto* mass_pointer = &simulation.particles.DataManager::get(AttributeName<T>("m"));
        for (auto& h : simulation.force->helpers)
            h->computePerNodeCNTolerance(simulation.particle_order, simulation.particle_group, simulation.block_offset,
                [&](int i, const Hessian& ddF, const T& ddJ, bool use_J) {
                    auto& Xp = simulation.particles.X[i];
                    auto& mass = (*mass_pointer)[i];
                    std::vector<TV> cached_w(27);
                    BSplineWeights<T, dim> spline(Xp, simulation.dx);
                    int cnt = 0;
                    simulation.grid.iterateKernel(spline, simulation.particle_base_offset[i], [&](const IV& node, const T& w, const TV& dw, GridState<T, dim>& g) {
                        if (g.idx < 0)
                            return;
                        nodeCNTol[g.idx] += w * mass * (use_J ? ddJ : ddF.norm());
                    });
                });
        tbb::parallel_for(0, (int)nodeCNTol.size(), [&](int nodeid) {
            nodeCNTol[nodeid] *= (eps * 24 * simulation.dx * simulation.dx * dt) / simulation.mass_matrix[nodeid];
            //if (nodeCNTol[nodeid] < eps)
            //    nodeCNTol[nodeid] = eps * 0.01;
        });
    }

    void matrixSanityCheck()
    {
        int n = simulation.num_nodes;
        using EIGEN_EXT::vec;
        using MATH_TOOLS::sqr;
        TVStack xx{ dim, n }, bb{ dim, n };
        bool isWrong = false;
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < dim; ++d) {
                xx.setZero();
                xx(d, i) = 1;
                /// matrix-free
                bb.setZero();
                // Note the relationship H dx = - df, where H is the stiffness matrix
                if (!simulation.quasistatic)
                    simulation.inertia->addScaledForceDifferential(-sqr(simulation.dt), xx, bb);
                simulation.addScaledForceDifferentials(-sqr(simulation.dt), xx, bb);
                /// matrix
                for (int r = 0; r < n; ++r) {
                    for (int idx = 0; idx < 125; ++idx) {
                        if (entryCol[r * 125 + idx] == i) {
                            if (entryVal[r * 125 + idx].norm() == 0) continue;
                            for (int dd = 0; dd < dim; ++dd) {
                                auto va = entryVal[r * 125 + idx](dd, d);
                                auto vb = bb(dd, r);
                                auto diff = std::abs(va - vb);
                                if (diff > 1e-10) {
                                    ZIRAN_INFO("mat[", r * dim + dd, "(", r, "), ", i * dim + d, "(", i, ")] is ", va, ", differs from the right one ", vb);
                                    //ZIRAN_INFO(std::abs(va - vb));
                                    isWrong = true;
                                    cnt++;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (isWrong)
            getchar();
    }

    void multiply(const TVStack& x, TVStack& b) const
    {
        ZIRAN_ASSERT(x.cols() == simulation.num_nodes);
        ZIRAN_ASSERT(b.cols() == simulation.num_nodes);
        using EIGEN_EXT::vec;
        using MATH_TOOLS::sqr;
        if (matrix_free) {
            b.setZero();
            // Note the relationship H dx = - df, where H is the stiffness matrix
            if (!simulation.quasistatic)
                simulation.inertia->addScaledForceDifferential(-sqr(simulation.dt), x, b);

            simulation.addScaledForceDifferentials(-sqr(simulation.dt), x, b);

            // for (auto& lf : simulation.forces) {
            //     lf->addScaledForceDifferential(-sqr(simulation.dt), x, b);
            // }
        }
        else {
            b.setZero();
            matrix.multiply(x, b);
        }
    }

    template <class Func>
    void setProjection(Func project_func)
    {
        project = project_func;
    }

    template <class Func>
    void setPreconditioner(Func preconditioner_func)
    {
        precondition = preconditioner_func;
    }
};
} // namespace ZIRAN

#endif