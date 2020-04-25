#ifndef LBFGS_H
#define LBFGS_H
#include <Ziran/CS/Util/Meta.h>
#include <Ziran/CS/Util/Timer.h>
#include <Ziran/Math/Geometry/PartioIO.h>
#include <iostream>

#include <unsupported/Eigen/ArpackSupport>
#include <Configurations.h>

namespace ZIRAN {
/**
  Limited memory BFGS

  Templatized on TOBJ the objective function
  which should implement the following functions
  void computeResidual(Vec&)
*/
template <class Objective>
class LBFGS {
    using Vec = typename Objective::NewtonVector;
    using T = typename Objective::Scalar;
    template <typename T, int Size>
    struct RingBuffer {
        std::array<T, Size> _buf;
        int _head, _tail, _size;
        RingBuffer()
            : _head{ 1 }, _tail{ 0 }, _size{ 0 } {}
        void push_back()
        {
            incTail();
            if (_size > Size) incHead();
        }
        void pop_back()
        {
            decTail();
        }
        T& back()
        {
            return _buf[_tail];
        }
        T& operator[](int index)
        {
            index += _head;
            index %= Size;
            return _buf[index];
        }
        int size()
        {
            return _size;
        }
        void incTail()
        {
            ++_tail, ++_size;
            if (_tail == Size) _tail = 0;
        }
        void decTail()
        {
            if (_size == 0) return;
            --_tail, --_size;
            if (_tail < 0) _tail = Size - 1;
        }
        void incHead()
        {
            if (_size == 0) return;
            ++_head, --_size;
            if (_head == Size) _head = 0;
        }
    };

    Objective& objective;

public:
    Vec residual, tmp;

    T tolerance;
    int max_iterations;
    T linear_solve_tolerance_scale;
    int timesSinceLastUpdate;

    LBFGS(Objective& objective, const T tolerance = (T)1e-6, const int max_iterations = 5)
        : objective(objective)
        , tolerance(tolerance)
        , max_iterations(max_iterations)
        , linear_solve_tolerance_scale((T)1)
        , timesSinceLastUpdate(0)
    {
    }

    T dotProduct(const Vec& A, const Vec& B)
    {
        return (A.array() * B.array()).sum();
    }

    void checkPreconditioningMatrix(Vec& x)
    {
        constexpr int dim = 3;
        using ColSpMat = Eigen::SparseMatrix<T, Eigen::ColMajor>;
        using ColSpIter = typename ColSpMat::InnerIterator;
        FILE* fout = fopen("mat.m", "w");
        Vec r, t;
        r.resizeLike(x);
        t.resizeLike(x);
        std::vector<Eigen::Triplet<T>> triplets;
        int n = r.cols();
        for (int i = 0; i < n; ++i) {
            ZIRAN_INFO("checking ", i, " of ", n);
            for (int d = 0; d < dim; ++d) {
                r.setZero();
                r(d, i) = 1;
                objective.precondition(r, t);
                for (int j = 0; j < n; ++j) {
                    for (int dd = 0; dd < dim; ++dd) {
                        //if (std::abs(t(dd, j)) > 1e-10)
                        triplets.emplace_back(j * dim + dd, i * dim + d, t(dd, j));
                        if (t(dd, j) != 0.0)
                            fprintf(fout, "%d %d %le\n", j * dim + dd + 1, i * dim + d + 1, t(dd, j));
                    }
                }
            }
        }
        fclose(fout);
        ColSpMat mat;
        mat.resize(n * dim, n * dim);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (typename ColSpMat::InnerIterator it(mat, k); it; ++it) {
                if (it.row() > it.col())
                    continue;
                auto val = it.value();
                auto symval = mat.coeffRef(it.col(), it.row());
                if (std::abs(val - symval) / std::abs(val) > 1e-3) {
                    ZIRAN_INFO("\t(", it.row(), ", ", it.col(), "): ", val, ", \t(", it.col(), ", ", it.row(), "): ", symval);
                }
            }
        }
        ZIRAN_INFO("finish symmetric check, press to begin positive-definite sanity check!");
        getchar();
    }

    void checkPreconditionMatrixPD(const Vec& x)
    {
        constexpr int dim = 3;
        int n = x.cols();
        T maxentry = 1e2;
        for (int j = 0; j < n; j++)
            for (int d = 0; d < dim; d++)
                if (x(d, j) > maxentry)
                    maxentry = x(d, j);

        Vec r, t, o;
        r.resizeLike(x);
        t.resizeLike(x);
        o.resizeLike(x);
        srand((unsigned)time(0));
        for (int i = 0; i < n / 200; ++i) {
            ZIRAN_INFO("checking ", i, "-th random vector");
            /// gen random vector
            for (int j = 0; j < n; j++)
                for (int d = 0; d < dim; ++d) {
                    T rnd = (T)rand() / (RAND_MAX);
                    r(d, j) = (rnd - 0.5) * maxentry;
                }
            o = r;
            objective.precondition(r, t);
            T ag = objective.innerProduct(o, t) / o.norm() / t.norm();
            if (ag <= 0) {
                ZIRAN_WARN("Testing random PD: occurs negative xAx^T ", ag, "!");
                getchar();
                return;
            }
        }
        ZIRAN_INFO("finishes random PD test for preconditioning matrix!");
        getchar();
    }

    void outputResidual(const std::string& fileNum, const Vec& residual)
    {
        outputParticlePos(fileNum);

        constexpr int dim = 3;
        using TV = Vector<T, dim>;
        using IV = Vector<int, dim>;

        StdVector<TV> gridPos(residual.cols());
        objective.simulation.grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            gridPos[g.idx] = node.template cast<T>() * objective.simulation.dx;
        });

        std::string filename = objective.simulation.output_dir.absolutePath(fileNum + ".bgeo");

        Partio::ParticlesDataMutable* parts = Partio::create();
        Partio::ParticleAttribute posH, pfH;
        posH = parts->addAttribute("position", Partio::VECTOR, 3);
        pfH = parts->addAttribute("pf", Partio::VECTOR, 1);

        // gather data as a flat array
        StdVector<T> pfData(residual.cols());
        for (unsigned int i = 0; i < residual.cols(); ++i) {
            pfData[i] = residual.col(i).norm();
        }

        // write to partio structure
        FILE* out = fopen(objective.simulation.output_dir.absolutePath(fileNum + ".txt").c_str(), "w");
        ZIRAN_ASSERT(out);
        for (unsigned int k = 0; k < residual.cols(); ++k) {
            int idx = parts->addParticle();
            float* posP = parts->dataWrite<float>(posH, idx);
            ;
            float* pfP = parts->dataWrite<float>(pfH, idx);

            for (int d = 0; d < 3; ++d)
                posP[d] = 0;
            for (int d = 0; d < dim; ++d)
                posP[d] = (float)gridPos[k][d];
            pfP[0] = (float)pfData[k];

            fprintf(out, "(%le, %le, %le): %le %le %le\n", gridPos[k][0], gridPos[k][1], gridPos[k][2], residual(0, k), residual(1, k), residual(2, k));
        }
        fclose(out);
        std::cout << "++++++++ residual output with norm = " << residual.norm() << std::endl;

        Partio::write(filename.c_str(), *parts);
        parts->release();
    }

    void outputParticlePos(const std::string& filename)
    {
        constexpr int dim = 3;
        using TV = Vector<T, dim>;
        using IV = Vector<int, dim>;

        std::string filepath = objective.simulation.output_dir.absolutePath(filename + ".obj");
        FILE* out = fopen(filepath.c_str(), "w");
        ZIRAN_ASSERT(out);

        // write to partio structure
        for (int k = 0; k < objective.simulation.particles.count; ++k) {
            fprintf(out, "v");
            for (int d = 0; d < dim; ++d) {
                fprintf(out, " %le", objective.simulation.particles.X.array[k](d));
            }

            if constexpr (dim == 2) {
                fprintf(out, " 0.0");
            }

            fprintf(out, "\n");
        }

        fclose(out);
    }

    void outputResidual_particle(const std::string& fileNum, const Vec& residual)
    {
        constexpr int dim = 3;
        using TV = Vector<T, dim>;
        using IV = Vector<int, dim>;

        StdVector<TV> particleRes(objective.simulation.particles.count, TV(0.0, 0.0, 0.0));
        tbb::parallel_for(0, (int)objective.simulation.particles.count, [&](int i) {
            TV& Xp = objective.simulation.particles.X.array[i];
            BSplineWeights<T, dim> spline(Xp, objective.simulation.dx);
            objective.simulation.grid.iterateKernel(spline, objective.simulation.particle_base_offset[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                particleRes[i] += w * residual.col(g.idx);
            });
        });

        std::string filename = objective.simulation.output_dir.absolutePath(fileNum + ".bgeo");

        Partio::ParticlesDataMutable* parts = Partio::create();
        Partio::ParticleAttribute posH, pfH;
        posH = parts->addAttribute("position", Partio::VECTOR, 3);
        pfH = parts->addAttribute("pf", Partio::VECTOR, 1);

        // gather data as a flat array
        StdVector<T> pfData(particleRes.size());
        for (unsigned int i = 0; i < particleRes.size(); ++i) {
            pfData[i] = particleRes[i].norm();
        }

        // write to partio structure
        for (unsigned int k = 0; k < particleRes.size(); ++k) {
            int idx = parts->addParticle();
            float* posP = parts->dataWrite<float>(posH, idx);
            ;
            float* pfP = parts->dataWrite<float>(pfH, idx);

            for (int d = 0; d < 3; ++d)
                posP[d] = 0;
            for (int d = 0; d < dim; ++d)
                posP[d] = (float)objective.simulation.particles.X.array[k](d);
            pfP[0] = (float)pfData[k];
        }

        Partio::write(filename.c_str(), *parts);
        parts->release();
    }

    bool solve(Vec& x, const bool verbose = false, const bool useLinesearch = false)
    {
        // outputParticlePos("particlePos");
        // exit(0);

        ZIRAN_TIMER();
        Timer t;
        T lowrankt = 0;

        residual.resizeLike(x);
        tmp.resizeLike(x);

        objective.updateState(x);
        objective.computeResidual(residual);

        // outputResidual(std::to_string(objective.simulation.getFrame()) + "_" +
        //   std::to_string(objective.simulation.getSubstep()) + "_0", residual);

        constexpr int historySize = 8;
        RingBuffer<Vec, historySize + 1> dxx, dg;
        RingBuffer<T, historySize + 1> dgTdx;
        for (int i = 0; i < historySize + 1; ++i) {
            dxx[i].resizeLike(x);
            dg[i].resizeLike(x);
        }
        dxx.push_back();
        dg.push_back();
        dgTdx.push_back();
        std::array<T, historySize> ksi;

        timesSinceLastUpdate = 0;
        auto shouldRebuildHessian = [](int it) {
            if (HOTSettings::useAdaptiveHessian)
                return (it & 0xf) == 0;
            else
                return it == 0;
        };
        for (int it = 0; it < max_iterations; it++, timesSinceLastUpdate++) {
            // T residual_norm = objective.computeNorm(residual);
            ZIRAN_INFO("\n\n");
            ZIRAN_INFO("LBFGS iter ", it);
            // ZIRAN_INFO("LBFGS iter ", it, "; (designated norm) residual = ", residual_norm);
            ZIRAN_CONTEXT("LBFGS step", it);
            // ZIRAN_CONTEXT(residual_norm);
            if (objective.shouldExitByCN(residual)) {
                ZIRAN_INFO("LBFGS terminates at ", it);
                // ZIRAN_VERB_IF(true, "LBFGS terminates at ", it, "; (designated norm) residual = ", residual_norm);
                ZIRAN_INFO("LBFGS lowrank update time: ", lowrankt, "s");
                return true;
            }

            if (shouldRebuildHessian(it)) {
                objective.HinvApproxInit();
                timesSinceLastUpdate = 0;
                while (dxx.size() > 0) {
                    dxx.pop_back();
                    dg.pop_back();
                    dgTdx.pop_back();
                }
                dxx.push_back();
                dg.push_back();
                dgTdx.push_back();
            }

            dg.back() = residual;

            t.start();
            ZIRAN_INFO("trace ", dxx.size() - 1, " (", dg.size() - 1, ") history residuals");
            for (int i = dxx.size() - 2; i >= 0; --i) {
                ksi[i] = dotProduct(dxx[i], residual) * dgTdx[i];
                residual -= ksi[i] * dg[i];
            }
            lowrankt += t.click(false).count();

            /// VAV cycle
            objective.precondition(residual, dxx.back());
            objective.project(dxx.back());
            if constexpr (false) {
                T ag = objective.angle(dxx.back(), residual);
                if (ag < 0) {
                    ZIRAN_WARN("detects irregular angle, enforce VAV cycle!");
                    objective.multiply(dxx.back(), tmp);
                    objective.project(tmp);
                    objective.precondition(tmp, dxx.back());
                    objective.project(dxx.back());
                }
            }

            t.start();
            ZIRAN_INFO("trace ", dxx.size() - 1, " history solutions");
            for (int i = 0; i < dxx.size() - 1; ++i)
                dxx.back() += dxx[i] * (ksi[i] - dotProduct(dg[i], dxx.back()) * dgTdx[i]);
            lowrankt += t.click(false).count();

            if (useLinesearch) {
                if (HOTSettings::debugMode > 0) {
                    T ag = objective.angle(dxx.back(), dg.back());
                    if (ag <= 0) {
                        ZIRAN_WARN("detects irregular angle, press to begin system matrix & preconditioning matrix SPD sanity check!");
                        objective.BCprojectionSanityCheck(dxx.back(), dg.back());
                        getchar();
                        objective.checkMultigridSystemMatrix();
                        ZIRAN_INFO("press to begin preconditioning matrix SPD sanity check!");
                        getchar();
                        if (HOTSettings::debugMode == 2)
                            checkPreconditioningMatrix(x);
                        else if (HOTSettings::debugMode == 1)
                            checkPreconditionMatrixPD(x);
                    }
                }
                objective.lineSearch(dxx.back(), residual, 1.0); ///< returns actual omega
            }
            objective.recoverSolution(dxx.back());
            x += dxx.back();
            objective.transformResidual(dxx.back());

            objective.updateState(x);
            objective.computeResidual(residual);

            // outputResidual(std::to_string(objective.simulation.getFrame()) + "_" +
            //   std::to_string(objective.simulation.getSubstep()) + "_" + std::to_string(it + 1), residual);

            t.start();
            dg.back() -= residual;
            dgTdx.back() = (T)1 / dotProduct(dg.back(), dxx.back());
            if (dgTdx.back() <= 0.0) {
                std::cout << "dropped" << std::endl;
                dxx.pop_back();
                dg.pop_back();
                dgTdx.pop_back();
            }
            dxx.push_back();
            dg.push_back();
            dgTdx.push_back();
            lowrankt += t.click(false).count();
        }
        return false;
    }
};
} // namespace ZIRAN
#endif
