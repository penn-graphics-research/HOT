#ifndef MULTIGRID_SIMULATION_FAST_H
#define MULTIGRID_SIMULATION_FAST_H

#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Physics/ConstitutiveModel/CorotatedIsotropic.h>
#include <Ziran/Math/Linear/Minres.h>
#include <MPM/MpmSimulationBase.h>
#include <Ziran/Math/Nonlinear/LBFGS.h>
#include <Ziran/Math/Nonlinear/ExtendedNewtonsMethod.h>

#include <tbb/concurrent_hash_map.h>
#include <Eigen/LU>
#include <Partio.h>
#include <map>
#include <mutex>
#include <Ziran/Sim/DiffTest.h>
#include <Ziran/Math/Linear/Minres.h>
#include <Ziran/Math/Geometry/CollisionObject.h>

#include "ImplicitSolver.h"
#include "Configurations.h"

namespace ZIRAN {

template <class T, int dim>
class MultigridSimulation : public MpmSimulationBase<T, dim> {
public:
    using Base = MpmSimulationBase<T, dim>;
    using Base::outputFileName;

    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    typedef Matrix<T, dim, dim> TM;
    typedef Matrix<T, dim, Eigen::Dynamic> TVStack;

    using Hessian = Eigen::Matrix<T, dim * dim, dim * dim>;
    using MPMSpMat = SparseMPMMatrix<T, int, dim>;
    using SpMat = SquareMatrix<T, dim>;
    using CollisionNodeArray = tbb::concurrent_vector<CollisionNode<T, dim>>;
    using ULL = unsigned long long;
    static constexpr ULL hash_seed = 100007;
    using Coord2IdMap = tbb::concurrent_hash_map<ULL, int>;
    using MGOperator = MultigridOperator<T, int, dim>;

    using Simulation = MultigridSimulation<T, dim>;
    using Objective = ImplicitSolverObjective<Simulation>;

    struct BaselineMultigrid {
        explicit BaselineMultigrid(StdVector<std::unique_ptr<MpmForceHelperBase<T, dim>>>& helpers)
            : force_helpers(helpers), particle_sorter(0), particle_base_offset(0), particle_order(0), particle_group(0), block_offset(0) {}
        auto& refCols() { return sysmat->_mat->entryCol; }
        auto& refVals() { return sysmat->_mat->entryVal; }
        StdVector<std::unique_ptr<MpmForceHelperBase<T, dim>>>& force_helpers;
        StdVector<uint64_t> particle_sorter;
        StdVector<uint64_t> particle_base_offset;
        StdVector<int> particle_order;
        std::vector<std::pair<int, int>> particle_group;
        std::vector<uint64_t> block_offset;
        MpmGrid<T, dim> grid;
        T dx;
        int num_nodes;
        Vector<T, Eigen::Dynamic> mass_matrix;
        CollisionNodeArray collision_nodes;
        std::unique_ptr<MPMSpMat> sysmat, promat;
        void clear()
        {
            //particle_sorter.clear();
            //particle_base_offset.clear();
            //particle_order.clear();
            //particle_group.clear();
            //block_offset.clear();
        }
    };
    std::vector<BaselineMultigrid> multigrids;

    Objective objective;
#if USE_GAST15_METRIC
    NewtonsMethod<Objective> newton;
#else
    ExtendedNewtonsMethod<Objective> newton;
#endif
    LBFGS<Objective> lbfgs;

    bool recomputeLocationBasedElementMeasure = false;

    void mark_colors(const std::vector<Vector<int, dim>>& id2coord, const std::unique_ptr<Coord2IdMap>& coord2id, std::vector<std::array<int, 3>>& colorOrder, std::array<std::vector<std::vector<int>>, 8>& coloredBlockDofs);
    void sortParticlesAndPolluteMultigrid(BaselineMultigrid& multigrid);
    void buildMultigridBoundaries(BaselineMultigrid& multigrid);
    void buildMultigridMatrices(BaselineMultigrid& multigrid, std::unique_ptr<SpMat>& resmat, std::unique_ptr<SpMat>& sysmat, std::unique_ptr<SpMat>& promat, std::unique_ptr<Coord2IdMap>& finerCoord2Id, std::vector<IV>& validFinerNodes);
    void particlesToMultigrids(MGOperator& mgp);

    MultigridSimulation()
        : Base()
        , objective(*this)
        , newton(objective, (T)1, 3)
        , lbfgs(objective, (T)1, 10000)
    {
    }

    void initialize()
    {
        Base::initialize();
        if (!Base::symplectic) {
            if (HOTSettings::systemBCProject && HOTSettings::boundaryType == 1) { ///< if no SLIP boundary at all, just stick with STICKY
                objective.initialize(
                    [&](TVStack& dv) {
                        for (auto iter = Base::collision_nodes.begin(); iter != Base::collision_nodes.end(); ++iter) {
                            int node_id = iter->node_id;
                            if (iter->shouldRotate)
                                dv(0, node_id) = 0;
                            else
                                dv.col(node_id).setZero();
                        }
                    });
            }
            else
                objective.initialize(
                    [&](TVStack& dv) {
                        for (auto iter = Base::collision_nodes.begin(); iter != Base::collision_nodes.end(); ++iter) {
                            int node_id = (*iter).node_id;
                            (*iter).project(dv.col(node_id));
                        }
                    });
        }
    }

    T computeCharacteristicNorm(T eps, T dt, int nodecnt, T& max_tol_p)
    {
        /// hack!
        static bool first = true;
        static double dPdFNorm = -1;
        static double dPdFNorm_max = -1;
        if (first) {
            using CI = CorotatedIsotropic<T, dim>;
            if (Base::particles.DataManager::exist(AttributeName<CI>(CI::name()))) {
                DisjointRanges subset(Base::particles.X.ranges, Base::particles.commonRanges(AttributeName<CI>(CI::name())));
                for (auto iter = Base::particles.subsetIter(subset, AttributeName<CI>(CI::name())); iter; ++iter) {
                    auto model = iter.template get<0>();
                    typename CI::Scratch s;
                    Eigen::Matrix<T, dim * dim, dim * dim> firstPiolaDerivative;
                    model.updateScratch(Eigen::Matrix<T, dim, dim>::Identity(), s);
                    model.firstPiolaDerivative(s, firstPiolaDerivative);
                    double curdPdFNorm = firstPiolaDerivative.norm();
                    if (dPdFNorm < 0 || curdPdFNorm < dPdFNorm)
                        dPdFNorm = curdPdFNorm;
                    if (dPdFNorm_max < 0 || curdPdFNorm > dPdFNorm_max)
                        dPdFNorm_max = curdPdFNorm;
                }
            }
        }
        if (dPdFNorm > 0) ///< probably there are no objects in the first frame at all
            first = false;
        objective.evaluatePerNodeCNTolerance(eps, dt);
        //return eps * dt * 24 * std::sqrt(n) * this->dx * this->dx * dPdFNorm;

        if (dPdFNorm_max != -1) { // computed something for it
            max_tol_p = eps * dt * 24 * std::sqrt(nodecnt) * this->dx * this->dx * dPdFNorm_max;
        }

        T cntol = eps * dt * 24 * std::sqrt(nodecnt) * this->dx * this->dx * dPdFNorm; ///< infinite norm
        if (cntol > 0)
            return cntol;
        return 0;
    }

    void startBackwardEuler()
    {
        Base::buildMassMatrix();
        objective.setPreconditioner([&](const TVStack& in, TVStack& out) {
            for (int i = 0; i < Base::num_nodes; i++) {
                for (int d = 0; d < dim; d++) {
                    out(d, i) = in(d, i) / Base::mass_matrix(i);
                }
            }
        });

        Base::dv.resize(dim, Base::num_nodes);
        Base::vn.resize(dim, Base::num_nodes);

        Base::buildInitialDvAndVnForNewton(); // This also builds collision_nodes
        // TODO: this should probably be done in objective reinitialize.
        // Which should be called at the beginning of newton.
        Base::force->backupStrain();
        objective.reinitialize(); // Reinitialize matrix sparsity pattern
    }

    void backwardEulerStep()
    {
        ZIRAN_TIMER();
        startBackwardEuler();
        for (auto f : Base::general_callbacks) {
            f(Base::frame, Base::BeforeNewtonSolve);
        }
        // run diff_test if we specify it
        if (Base::diff_test)
            runDiffTest<T, dim, Objective>(Base::num_nodes, Base::dv, objective, Base::diff_test_perturbation_scale);

        T maxcntol = -1;

        if (HOTSettings::useCN) {
            computeCharacteristicNorm(HOTSettings::cneps, this->dt, Base::dv.cols(), maxcntol);
            //printf("characteristic norm tolerance %e (eps: %e, dt: %e, dx: %e, n: %d)\n", HOTSettings::characterNorm, HOTSettings::cneps, this->dt, this->dx, (int)objective.id2coord.size());
            newton.tolerance = maxcntol;
            lbfgs.tolerance = maxcntol;
            objective.minres.setTolerance(maxcntol);
            objective.cg.setTolerance(maxcntol);
        }
        else {
            lbfgs.tolerance = newton.tolerance = HOTSettings::cneps;
        }
        ZIRAN_INFO(std::scientific, "maximum characteristic norm tolerance ", maxcntol, "(eps:", HOTSettings::cneps, ",n:", (int)Base::dv.cols(), ".  max cntol =  ", maxcntol);
        ZIRAN_INFO(std::scientific, "newton tol:", newton.tolerance, "; lbfgs tol:", lbfgs.tolerance, "; minres tol:", objective.minres.tolerance, "(", objective.minres.relative_tolerance, "), cg tol:", objective.cg.tolerance, "(", objective.cg.relative_tolerance, ")");

        objective.isNewStep = true;
        if (HOTSettings::linesearch)
            objective.resetLSFlag(Base::dv);
        if (HOTSettings::lsolver != 3)
            newton.solve(Base::dv, Base::verbose);
        else
            lbfgs.solve(Base::dv, Base::verbose, HOTSettings::linesearch);

        for (auto f : Base::general_callbacks) {
            f(Base::frame, Base::AfterNewtonSolve);
        }
        // run diff_test if we specify it
        if (Base::diff_test)
            runDiffTest<T, dim, Objective>(Base::num_nodes, Base::dv, objective, Base::diff_test_perturbation_scale);

        Base::force->restoreStrain();

        Base::constructNewVelocityFromNewtonResult();
    }

    virtual void advanceOneTimeStep(double dt)
    {
        ZIRAN_TIMER();
        ZIRAN_INFO("Advance one time step from time ", std::setw(7), Base::step.time, " with                     dt = ", dt);

        Base::reinitialize();

        if (Base::step.time == 0 && recomputeLocationBasedElementMeasure) {
            auto& Xarray = Base::particles.X.array;
            auto& marray = Base::particles.mass.array;
            auto* vol_pointer = &Base::particles.DataManager::get(element_measure_name<T>());
            for (uint64_t color = 0; color < (1 << dim); ++color) {
                tbb::parallel_for(0, (int)Base::particle_group.size(), [&](int group_idx) {
                    if ((Base::block_offset[group_idx] & ((1 << dim) - 1)) != color)
                        return;
                    for (int idx = Base::particle_group[group_idx].first; idx <= Base::particle_group[group_idx].second; ++idx) {
                        int i = Base::particle_order[idx];
                        TV& Xp = Xarray[i];
                        T mass = marray[i];
                        BSplineWeights<T, dim> spline(Xp, Base::dx);
                        Base::grid.iterateKernel(spline, Base::particle_base_offset[i], [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                            g.m += w * mass;
                        });
                    }
                });
            }
            Base::grid.getNumNodes();
            // store rho as mass
            Base::grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
                g.m /= std::pow(Base::dx, (T)dim);
            });

            tbb::parallel_for(0, (int)Base::particles.count, [&](int i) {
                TV& Xp = Xarray[i];
                T& mass = marray[i];
                T rho = 0;
                BSplineWeights<T, dim> spline(Xp, Base::dx);
                Base::grid.iterateKernel(spline, Base::particle_base_offset[i], [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                    rho += w * g.m;
                });
                (*vol_pointer)[i] = mass / rho;
            });

            Base::grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
                g.m = (T)0;
            });
        }

        for (auto f : Base::before_p2g_callbacks) {
            f();
        }
        Base::particlesToGrid();
        for (auto f : Base::before_euler_callbacks) {
            f();
        }
        backwardEulerStep();
        ZIRAN_INFO("After BE (but before G2P), updating collision object to time ", Base::step.time + dt);
        for (size_t k = 0; k < Base::collision_objects.size(); ++k)
            if (Base::collision_objects[k]->updateState) {
                Base::collision_objects[k]->updateState(Base::step.time + dt, *Base::collision_objects[k]);
            }
        Base::gridToParticles(dt);
    }

    const char* name() override { return "multigrid"; }
};
} // namespace ZIRAN

#include "MultigridSimulation.inl"

#endif
