#ifndef MULTIGRID_PRECONDITIONER_H
#define MULTIGRID_PRECONDITIONER_H
#include "MPMMultigridMatrix.h"
#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/KrylovSolvers.h>
#include <Ziran/Math/Linear/Minres.h>
#include <Ziran/Math/Linear/GeneralizedMinimalResidual.h>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Sim/SimulationBase.h>
#include <Ziran/Math/Linear/EigenSparseLU.h>
#include <vector>
#include <map>
#include "function_ref.hpp"
#include "Timer.hpp"
//#include <Ziran/CS/Util/Timer.h>
#include "Configurations.h"

namespace ZIRAN {

enum class SmootherType { Jacobi,
    GS,
    CG,
    LBFGS,
    Minres,
    Total };

template <class T, class IndexType, int dim>
struct MultigridOperator {

    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using Vec = Vector<T, Eigen::Dynamic>;
    using MPMSpMat = SparseMPMMatrix<T, IndexType, dim>;
    using SpMat = SquareMatrix<T, dim>;
    template <typename ScalerFunc>
    struct Operator {
        const MPMSpMat& A;
        ScalerFunc scaler_func;
        Operator(const MPMSpMat& A_, ScalerFunc scaler_func)
            : A(A_), scaler_func(scaler_func) {}
        void multiply(const TVStack& in, TVStack& out) const
        {
            A.multiply(in, out);
        }
        void project(TVStack&) const {}
        void precondition(const TVStack& in, TVStack& out) const
        {
            scaler_func(in, out, A);
        }
    };

    inline static int goingup{ 1 };
    std::vector<IndexType> dofs;
    std::vector<std::unique_ptr<MPMSpMat>> sysmats;
    std::vector<std::unique_ptr<MPMSpMat>> promats;
    inline static std::vector<Vec> nodeCNTols;
    inline static std::vector<Vec> nodeMasses;
    inline static std::vector<TVStack> initialResiduals;
    std::vector<TVStack> residuals, dAus;
    std::vector<TVStack> sols, dus;
    inline static std::vector<TVStack> tmps;

    inline static int level; ///< curent level

    bool updateResidual; ///< update residual after each calling
    inline static void (*scaler_func)(const TVStack&, TVStack&, const MPMSpMat&);
    struct {
        //std::function<void(TVStack&, TVStack&, TVStack&, TVStack&, MPMSpMat&, int, T)> smoothFunc;
        void (*smoothFunc)(TVStack&, TVStack&, TVStack&, TVStack&, MPMSpMat&, int, T);
        std::function<T(const TVStack&)> normFunc;
        std::function<T(int)> tolFunc;
    } regular, top;
    int splitLevel;
    std::function<int(int)> downIterFunc;
    std::function<int(int)> topIterFunc;
    std::function<int(int)> upIterFunc;

    std::function<void(TVStack&)> correctResidualProjection;

    MultigridOperator() = default;
    ~MultigridOperator() = default;
    void init(
        std::vector<IndexType>& dofs_,
        std::vector<std::unique_ptr<SpMat>>& resmats_,
        std::vector<std::unique_ptr<SpMat>>& sysmats_,
        std::vector<std::unique_ptr<SpMat>>& promats_)
    {
        dofs = std::move(dofs_);
        sysmats.clear();
        promats.clear();
        residuals.clear();
        initialResiduals.clear();
        sols.clear();
        dAus.clear();
        dus.clear();
        tmps.clear();
        for (int i = 0; i < (int)dofs.size() - 1; ++i) {
            ZIRAN_INFO("\tLevel ", i, " has ", dofs[i], " dofs");
            sysmats.push_back(std::move(std::make_unique<MPMSpMat>(sysmats_[i])));
            //printf("\npro colsize: %d\t res colsize: %d\n\n", promats_[i]->colsize, resmats_[i]->colsize);
            promats.push_back(std::move(std::make_unique<MPMSpMat>(resmats_[i], promats_[i])));
            residuals.emplace_back(dim, dofs[i]);
            initialResiduals.emplace_back(dim, dofs[i]);
            sols.emplace_back(dim, dofs[i]);
            dAus.emplace_back(dim, dofs[i]);
            dus.emplace_back(dim, dofs[i]);
            tmps.emplace_back(dim, dofs[i]);
        }
        ZIRAN_INFO("\tLevel ", dofs.size() - 1, " has ", dofs.back(), " dofs");
        sysmats.push_back(std::move(std::make_unique<MPMSpMat>(sysmats_.back())));
        residuals.emplace_back(dim, dofs.back());
        initialResiduals.emplace_back(dim, dofs.back());
        sols.emplace_back(dim, dofs.back());
        dAus.emplace_back(dim, dofs.back());
        dus.emplace_back(dim, dofs.back());
        tmps.emplace_back(dim, dofs.back());
    }

    void checkSystemMatrix()
    {
        for (int i = 0; i < (int)dofs.size(); ++i) {
            ZIRAN_INFO("checking level (symmetric) ", i + 1, " of ", (int)dofs.size());
            sysmats[i]->_mat->symmetricSanityCheck();
            ZIRAN_INFO("checking PD");
            sysmats[i]->_mat->SPDSanityCheck();
        }
    }

    static auto l2norm(const TVStack& r)
    {
        T norm_sq = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, r.cols(), 256), (T)0,
            [&](const tbb::blocked_range<int>& range, T ns) -> T {
                int start = range.begin();
                int length = (range.end() - range.begin());
                const auto& r_block = r.middleCols(start, length);
                return ns + ((r_block.colwise().squaredNorm()).array()).sum();
            },
            [](T a, T b) -> T { return a + b; });
        return std::sqrt(norm_sq);
    }
    static void scale_diagonal_entry_inverse(const TVStack& r, TVStack& mr, const MPMSpMat& A)
    {
        tbb::parallel_for(IndexType(0), (IndexType)r.cols(), [&](IndexType i) {
            mr.col(i) = A._mat->diagonalEntry[i] * r.col(i);
        });
    };
    static void scale_diagonal_block_inverse(const TVStack& r, TVStack& mr, const MPMSpMat& A)
    {
        tbb::parallel_for(IndexType(0), (IndexType)r.cols(), [&](IndexType i) {
            mr.col(i) = A._mat->diagonalBlock[i] * r.col(i);
        });
    };
    static auto dotProduct(const TVStack& A, const TVStack& B)
    {
        return (A.array() * B.array()).sum();
    }

    static void jacobi_smooth(TVStack& u, TVStack& r, TVStack& du, TVStack& dAu, MPMSpMat& A, int iterations, T tolerance)
    {
        int cnt = 0;
        for (; iterations--; ++cnt) {
            scaler_func(r, du, A);
            du *= HOTSettings::topomega;
            u += du;
            A.multiply(du, dAu);
            A.project(dAu);
            r -= dAu;
            //if (HOTSettings::revealJacobi) printf("\t%d-th jacobi iter residual norm: %f\n", cnt++, norm(r));
        }
        ZIRAN_INFO("JACOBI smoother exits at ", cnt, " iters");
    }
    static void optimal_jacobi_smooth(TVStack& u, TVStack& r, TVStack& du, TVStack& dAu, MPMSpMat& A, int iterations, T tolerance)
    {
        T omega;
        int cnt = 0;
        for (; iterations--; ++cnt) {
            if (l2norm(r) < tolerance) break;
            scaler_func(r, du, A);
            A.multiply(du, dAu);
            A.project(dAu);
            omega = dotProduct(du, r) / dotProduct(du, dAu);
            u += du * omega;
            r -= dAu * omega;
            //if (HOTSettings::revealJacobi) printf("\t%d-th jacobi iter residual norm: %f\n", cnt++, norm(r));
        }
        ZIRAN_INFO("Optimal JACOBI smoother exits at ", cnt, " iters");
    }
    static void cg_smooth(TVStack& u, TVStack& r, TVStack& du, TVStack& dAu, MPMSpMat& A, int iterations, T tolerance)
    {
        TVStack& z = tmps[level];
        //z.resizeLike(u);
        T omega, infnorm;

        //tolerance *= A._mat->rows() * 64*64;
        scaler_func(initialResiduals[level], z, A);
        T zTrk0 = dotProduct(z, initialResiduals[level]);

        scaler_func(r, z, A);
        du = z;
        T zTrk = dotProduct(z, r);
        double cgratio =
#if 0
            std::min((T)0.5, std::sqrt(std::max(std::sqrt(zTrk0), HOTSettings::cgratio)));
#else
            0.5; //HOTSettings::cgratio;
#endif
            tolerance = zTrk0 * cgratio * cgratio; //0.25;
        // tolerance = zTrk0 * 0.25;
        int cnt = 0;
        for (; iterations--;) {
            if (zTrk < tolerance) break;
            A.multiply(du, dAu);
            A.project(dAu);
            omega = zTrk / dotProduct(dAu, du);
            u += du * omega;
            r -= dAu * omega;
            scaler_func(r, z, A);
            T zTrkPre = zTrk;
            T beta = (zTrk = dotProduct(z, r)) / zTrkPre;
            du = z + beta * du;
            ++cnt;
        }
        ZIRAN_INFO("PCG smoother exits at ", cnt, " iters. l2 norm sqr is ", zTrk, ". tolerance sqr is ", tolerance);
    }
    static void chebyshev_smooth(TVStack& u, TVStack& r, TVStack& du, TVStack& dAu, MPMSpMat& A, int iterations, T tolerance)
    {
        TVStack& p = tmps[level];
        //p.resizeLike(du);
        T d = (A._mat->lMax + A._mat->lMin) / 2;
        T c = (A._mat->lMax - A._mat->lMin) / 2;

        //ZIRAN_INFO("used eig val max: ", A._mat->lMax, ", eig val min: ", A._mat->lMin);
        // only activate when used as solver
        //scaler_func(initialResiduals[level], p, A);
        //tolerance = dotProduct(p, initialResiduals[level]) * 0.25;
        int cnt = 1;
        iterations--;
        /// first iteration
        scaler_func(r, p, A);
        T alpha = 1 / d, beta;
        du = p;
        A.multiply(du, dAu);
        A.project(dAu);
        u += (alpha * du);
        r -= (alpha * dAu);

        for (; iterations-- > 0; ++cnt) {
            scaler_func(r, p, A);
            beta = 0.5 * c * c * alpha * alpha;
            if (cnt > 1)
                beta *= 0.5;
            alpha = 1 / (d - beta / alpha);
            du = p + beta * du;

            A.multiply(du, dAu);
            A.project(dAu);
            u += (alpha * du);
            r -= (alpha * dAu);
        }
        //
        ZIRAN_INFO("Chebyshev smoother exits at ", cnt, " iters");
    }

    static void gs_smooth(TVStack& u, TVStack& r, TVStack& du, TVStack& dAu, MPMSpMat& A, int iterations, T tolerance)
    {
        int cnt = 0;
        auto& mat = *(A._mat);
        TVStack& hdu = tmps[level];
        //hdu.resizeLike(du);
        bool calced = false;
        iterations = ((iterations + 1) >> 1);
        for (; iterations--;) {
            hdu.setZero();
            for (int c = 0; c < (1 << dim); ++c)
                tbb::parallel_for(0, (int)mat.coloredBlockDofs[c].size(), [&](int bid) {
                    const auto& blockNodes = mat.coloredBlockDofs[c][bid];
                    for (int ii = 0; ii < (int)blockNodes.size(); ++ii) {
                        int i = blockNodes[ii];
                        Eigen::Matrix<T, dim, 1> sigma;
                        for (int d = 0; d < dim; ++d)
                            sigma[d] = 0;
                        typename MPMSpMat::SpIter it(mat, i);
                        for (; it(); ++it) {
                            if (mat.comp(mat.colorOrder[it.col()], mat.colorOrder[i]) < 0)
                                sigma += it.value() * hdu.col(it.col());
                        }
                        hdu.col(i) = mat.diagonalBlock[i] * (r.col(i) - sigma);
                    }
                });
            for (int i = 0; i < r.cols(); ++i)
                hdu.col(i) = mat.diagonalVal[i] * hdu.col(i);

            du.setZero();
            for (int c = (1 << dim) - 1; c >= 0; --c)
                tbb::parallel_for(0, (int)mat.coloredBlockDofs[c].size(), [&](int bid) {
                    const auto& blockNodes = mat.coloredBlockDofs[c][bid];
                    for (int ii = blockNodes.size() - 1; ii >= 0; --ii) {
                        int i = blockNodes[ii];
                        Eigen::Matrix<T, dim, 1> sigma;
                        for (int d = 0; d < dim; ++d)
                            sigma[d] = 0;
                        typename MPMSpMat::SpIter it(mat, i);
                        for (; it(); ++it)
                            if (mat.comp(mat.colorOrder[it.col()], mat.colorOrder[i]) > 0)
                                sigma += it.value() * du.col(it.col());
                        du.col(i) = mat.diagonalBlock[i] * (hdu.col(i) - sigma);
                    }
                });
            u += du;
            A.multiply(du, dAu);
            A.project(dAu);
            r -= dAu;
            ++cnt;
        }
        ZIRAN_INFO("symmetric GS smoother exits at ", cnt, " iters");
    }

    static void IC_smooth(TVStack& u, TVStack& r, TVStack& du, TVStack& dAu, MPMSpMat& A, int iterations, T tolerance)
    {
        A._mat->solveIC(r, u);
        ZIRAN_INFO("Finish incomplete cholesky solve!");

        /*TVStack& z = tmps[level];
        //z.resizeLike(u);
        T omega, infnorm;

        //tolerance *= A._mat->rows() * 64*64;
        A._mat->solveIC(initialResiduals[level], z);
        //scaler_func(initialResiduals[level], z, A);
        T zTrk0 = dotProduct(z, initialResiduals[level]);

        A._mat->solveIC(r, z);
        //scaler_func(r, z, A);
        du = z;
        T zTrk = dotProduct(z, r);
        tolerance = zTrk0 * 0.25;
        int cnt = 0;
        for (; iterations--;) {
            if (zTrk < tolerance) break;
            A.multiply(du, dAu);
            A.project(dAu);
            omega = zTrk / dotProduct(dAu, du);
            u += du * omega;
            r -= dAu * omega;
            A._mat->solveIC(r, z);
            //scaler_func(r, z, A);
            T zTrkPre = zTrk;
            T beta = (zTrk = dotProduct(z, r)) / zTrkPre;
            du = z + beta * du;
            ++cnt;
            
            //DEBUG output quadratic form and residual
            ZIRAN_INFO("ICPCG smoother iter", cnt, ", res = ", zTrk);
            A.multiply(u, dAu);
            ZIRAN_INFO("energy = ", 0.5 * dotProduct(dAu, u) - dotProduct(u, initialResiduals[level]));
        }
        ZIRAN_INFO("ICPCG smoother exits at ", cnt, " iters. l2 norm sqr is ", zTrk, ". tolerance sqr is ", tolerance);*/
    }

    void operator()(const TVStack& in, TVStack& out)
    {
        //ZIRAN_WARN("Multigrid Preconditioning");
        ZIRAN_TIMER();
        int levelCnt = sysmats.size();

        Timer t;
        T timePerLevel[10][4]; ///< smooth, restrict, prolongate, merge
        ZIRAN_ASSERT(levelCnt <= 10, "Level depth exceeds 10! Too Deep!");
        for (int i = 0; i < levelCnt; ++i)
            for (int j = 0; j < 4; ++j)
                timePerLevel[i][j] = 0;

        residuals[0] = in;
        correctResidualProjection(residuals[0]); ///< BC project residual locally for AMG
        out.setZero();

        if (levelCnt > 1)
            promats[0]->transposeMultiply(residuals[0], initialResiduals[1]);
        else
            initialResiduals[0] = residuals[0];
        for (int l = 1; l < levelCnt - 1; ++l)
            promats[l]->transposeMultiply(initialResiduals[l], initialResiduals[l + 1]);

        for (level = 0; level < levelCnt - 1; ++level) {
            TVStack& sol = level == 0 ? out : sols[level];

            t.start();
            (level < splitLevel ? regular : top).smoothFunc(sol, residuals[level], dus[level], dAus[level], *sysmats[level], level < splitLevel ? upIterFunc(level) : topIterFunc(level), (level < splitLevel ? regular : top).tolFunc(level));
            timePerLevel[level][0] += t.click(false).count();

            t.start();
            promats[level]->transposeMultiply(residuals[level], residuals[level + 1]);
            timePerLevel[level][1] += t.click(false).count();
            sols[level + 1].setZero(dim, dofs[level + 1]);
        }
        t.start();
        top.smoothFunc(level == 0 ? out : sols[level], residuals[level], dus[level], dAus[level], *sysmats[level], topIterFunc(level), top.tolFunc(level));
        timePerLevel[level][0] += t.click(false).count();
        for (--level; level >= 0; --level) {
            TVStack& sol = level == 0 ? out : sols[level];
            t.start();
            promats[level]->multiply(sols[level + 1], dus[level]);
            timePerLevel[level][2] += t.click(false).count();

            t.start();
            sol += dus[level];
            sysmats[level]->multiply(dus[level], dAus[level]);
            residuals[level] -= dAus[level];
            timePerLevel[level][3] += t.click(false).count();

            t.start();
            (level < splitLevel ? regular : top).smoothFunc(sol, residuals[level], dus[level], dAus[level], *sysmats[level], level < splitLevel ? downIterFunc(level) : topIterFunc(level), (level < splitLevel ? regular : top).tolFunc(level));
            timePerLevel[level][0] += t.click(false).count();
        }
        ZIRAN_INFO("V-Cycle Timing Breakdown.\n\t[smooth]\t\t[restrict]\t\t[prolongate]\t\t[merge]\n");
        for (int i = 0; i < levelCnt; ++i)
            ZIRAN_INFO(i, "\t", timePerLevel[i][0], "\t\t", timePerLevel[i][1], "\t\t", timePerLevel[i][2], "\t\t", timePerLevel[i][3]);
        return;
    };
};

template <class T, int dim>
constexpr auto quadratic_weight_template()
{
    constexpr int locCnt = pow(2, dim);
    constexpr int neighborCnt = pow(3, dim);
    std::array<std::array<T, pow(3, dim)>, locCnt> wts{};
    T w1d[2][3] = { 0.125, 0.75, 0.125, 0, 0.5, 0.5 };
    for (int loc = 0; loc < locCnt; ++loc) {
        int loc_offsets[3]{ 0, 0, 0 };
        for (int d = 0, l = loc; d < dim; ++d, l >>= 1)
            loc_offsets[dim - d - 1] = l & 1;
        for (int neighbor = 0; neighbor < neighborCnt; ++neighbor) {
            T weight = 1.;
            for (int d = 0, nid = neighbor; d < dim; ++d, nid /= 3)
                weight *= w1d[loc_offsets[dim - d - 1]][nid % 3];
            wts[loc][neighbor] = weight;
        }
    }
    return wts;
}

template <class T, int dim>
constexpr auto linear_weight_template()
{
    constexpr int locCnt = pow(2, dim);
    constexpr int neighborCnt = pow(3, dim);
    std::array<std::array<T, pow(3, dim)>, locCnt> wts{};
    T w1d[2][3] = { { 0.0, 1.0, 0.0 }, { 0, 0.5, 0.5 } };
    for (int loc = 0; loc < locCnt; ++loc) {
        int loc_offsets[3]{ 0, 0, 0 };
        for (int d = 0, l = loc; d < dim; ++d, l >>= 1)
            loc_offsets[dim - d - 1] = l & 1;
        //ZIRAN_INFO("\tloc: ", loc_offsets[0], ", ", loc_offsets[1], ", ", loc_offsets[2]);
        for (int neighbor = 0; neighbor < neighborCnt; ++neighbor) {
            T weight = 1.;
            for (int d = 0, nid = neighbor; d < dim; ++d, nid /= 3)
                weight *= w1d[loc_offsets[dim - d - 1]][nid % 3];
            //ZIRAN_INFO(neighbor / 9, ", ", (neighbor / 3) % 3, ", ", neighbor % 3, " : ", weight);
            wts[loc][neighbor] = weight;
        }
    }
    return wts;
}

template <class T, class IndexType, int dim>
struct MultigridBuilder {
    using TV = Vector<T, dim>;
    using IV = Vector<IndexType, dim>;
    using TM = Matrix<T, dim, dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using Vec = Vector<T, Eigen::Dynamic>;
    //using IVec = Vector<IndexType, Eigen::Dynamic>;
    using IVec = std::vector<IV>;
    using MPMSpMat = SparseMPMMatrix<T, IndexType, dim>;
    using SpMat = SquareMatrix<T, dim>;

    template <typename Object>
    void setup_logic(Object& obj, Vec& mass, int regularSmootherOpt, int topSmootherOpt, int scalerOpt)
    {
        //void (*scaler)(const TVStack&, TVStack&, const MPMSpMat&){ nullptr };
        //std::function<void(const TVStack&, TVStack&, const MPMSpMat&)> scaler;
        switch (scalerOpt) {
        case 0:
            Object::scaler_func = Object::scale_diagonal_entry_inverse;
            break;
        case 1:
            Object::scaler_func = Object::scale_diagonal_block_inverse;
            break;
        default:
            ZIRAN_ASSERT(false, "The Dinv function picked doesn't exist.");
            break;
        }
        static auto selectSmoother = [](auto& smoother, int opt) {
            switch (opt) {
            case 0:
                smoother = Object::jacobi_smooth;
                break;
            case 1:
                smoother = Object::optimal_jacobi_smooth;
                break;
            case 2:
                smoother = Object::cg_smooth;
                break;
            case 5:
                smoother = Object::gs_smooth;
                break;
            case 6:
                smoother = Object::chebyshev_smooth;
                break;
            case 7:
                smoother = Object::IC_smooth;
                break;
            default:
                ZIRAN_ASSERT(false, "No proper smoother is selected!");
            }
        };
        selectSmoother(obj.regular.smoothFunc, regularSmootherOpt);
        selectSmoother(obj.top.smoothFunc, topSmootherOpt);
    }

    template <typename Object>
    void setup_parameters(Object& obj, int times, int levelscale, T localTolerance)
    {
        //obj.top.tolFunc = obj.regular.tolFunc = [localTolerance](int level) { return localTolerance; };
        //obj.top.tolFunc = [](int level) { return HOTSettings::characterNorm; };
        obj.top.tolFunc = [](int level) { return HOTSettings::cneps * HOTSettings::cneps; };
        //if (HOTSettings::levelCnt == 1)
        //    obj.top.tolFunc = [](int level) { return 1e-14; };
        obj.regular.tolFunc = [](int level) { return 0; }; /// small enough to ensure smooth times
        obj.downIterFunc = [times, levelscale](int level) { return times + level * levelscale; };
        if (HOTSettings::topDownMGS) {
            obj.splitLevel = 1;
            obj.upIterFunc = [](int level) { return 0; };
            obj.topIterFunc = [](int level) { return 10000; };
        }
        else {
            obj.splitLevel = HOTSettings::levelCnt - 1;
            obj.upIterFunc = [times, levelscale](int level) { return times + level * levelscale; };
            if (HOTSettings::levelCnt == 1)
                obj.topIterFunc = obj.upIterFunc;
            else {
                if (!(HOTSettings::coarseSolver == 2 || HOTSettings::coarseSolver == 6))
                    obj.topIterFunc = [times, levelscale](int level) { return (times + level * levelscale) * 3; };
                else
                    obj.topIterFunc = [](int level) { return 10000; };
            }
        }
    }

    template <typename Object>
    void build(Object& obj, Vec& mass, std::function<void(TVStack&)>& project, const TVStack& dRhs, int levelCnt, std::vector<IV>& id2coord, std::vector<int>& entryCol, std::vector<TM>& entryVal, const Vec& nodeCNTol)
    {
        ZIRAN_TIMER();
        ZIRAN_INFO("=============Start building===========");
        using ULL = unsigned long long;
        constexpr ULL hash_seed = 100007;
#define kernel_range 2
#if kernel_range == 2
        auto wts = linear_weight_template<T, dim>();
#elif kernel_range == 3
        auto wts = quadratic_weight_template<T, dim>();
#else
        ZIRAN_ASSERT(false);
#endif
        std::vector<int> dofs;
        std::vector<std::unique_ptr<SpMat>> resmats;
        std::vector<std::unique_ptr<SpMat>> sysmats;
        std::vector<std::unique_ptr<SpMat>> promats;

        sysmats.resize(levelCnt);
        promats.resize(levelCnt - 1);
        resmats.resize(levelCnt - 1);

        sysmats[0] = std::make_unique<SpMat>();
        sysmats[0]->colsize = pow(5, dim);
        sysmats[0]->entryCol = entryCol;
        sysmats[0]->entryVal = entryVal;
        sysmats[0]->buildDiagonal(HOTSettings::Ainv);
        static auto markColors = [&id2coord](std::vector<std::array<int, 3>>& colorOrder, std::array<std::vector<std::vector<int>>, 8>& coloredBlockDofs) {
            std::array<int, 8> blockCnts;
            for (auto& bc : blockCnts)
                bc = 0;
            std::array<std::unordered_map<ULL, int>, 8> blockIds;
            for (int i = 0; i < (int)id2coord.size(); ++i) {
                int blockIndices[dim];
                for (int d = 0; d < dim; ++d)
                    blockIndices[d] = id2coord[i][d] >> 2;
                int color = ((blockIndices[0] & 1) << 2) | ((blockIndices[1] & 1) << 1) | (blockIndices[2] & 1);
                ULL hash_key = (ULL)blockIndices[0] * hash_seed * hash_seed + (ULL)blockIndices[1] * hash_seed + (ULL)blockIndices[2];
                int blockId;
                if (blockIds[color].find(hash_key) == blockIds[color].end()) {
                    blockIds[color][hash_key] = blockCnts[color]++;
                    coloredBlockDofs[color].emplace_back();
                }
                blockId = blockIds[color][hash_key];
                auto& blockNodes = coloredBlockDofs[color];
                blockNodes[blockId].push_back(i);
                colorOrder[i][0] = color;
                colorOrder[i][1] = blockId;
                colorOrder[i][2] = blockNodes[blockId].size();
            }
        };
        if (HOTSettings::coarseSolver == 5 || HOTSettings::smoother == 5) {
            sysmats[0]->allocateColorIds();
            markColors(sysmats[0]->colorOrder, sysmats[0]->coloredBlockDofs);
        }
        if ((HOTSettings::coarseSolver == 6 && HOTSettings::levelCnt == 1) || (HOTSettings::smoother == 6 && HOTSettings::levelCnt > 1))
            sysmats[0]->estimate2norm();
        if ((HOTSettings::coarseSolver == 7 && HOTSettings::levelCnt == 1))
            sysmats[0]->setupICSolver();
        ZIRAN_ASSERT(!(HOTSettings::smoother == 7 && HOTSettings::levelCnt > 1), "Do not use IC solver as smoother in the multigrid!");

        dofs.clear();
        dofs.emplace_back((int)id2coord.size());

        for (int level = 0; level < levelCnt - 1; ++level) {
            // ZIRAN_INFO("\tBuilding Level ", level);
            std::vector<IV> new_id2coord;
            // simple mapping coord2id
            std::unordered_map<ULL, int> new_coord2id;

            auto& promat = promats[level];
            promat = std::make_unique<SpMat>();
            promat->colsize = pow(kernel_range, dim);
            promat->entryCol.resize(id2coord.size() * pow(kernel_range, dim));
            promat->entryVal.resize(id2coord.size() * pow(kernel_range, dim));
            for (int i = 0; i < (int)id2coord.size(); ++i) {
                const IV& coord = id2coord[i];
                int x = coord(0);
                int y = coord(1);
                int z = coord(2);
                const auto& wt = wts[(x & 1) * 4 + (y & 1) * 2 + (z & 1)];
#if kernel_range == 2
                for (int new_x = x / 2; new_x <= x / 2 + 1; ++new_x)
                    for (int new_y = y / 2; new_y <= y / 2 + 1; ++new_y)
                        for (int new_z = z / 2; new_z <= z / 2 + 1; ++new_z) {
#else
                for (int new_x = x / 2 - 1; new_x <= x / 2 + 1; ++new_x)
                    for (int new_y = y / 2 - 1; new_y <= y / 2 + 1; ++new_y)
                        for (int new_z = z / 2 - 1; new_z <= z / 2 + 1; ++new_z) {
#endif
                            int idx = (new_x - x / 2 + 1) * 9 + (new_y - y / 2 + 1) * 3 + new_z - z / 2 + 1;
#if kernel_range == 2
                            int linear_idx = (new_x - x / 2) * 4 + (new_y - y / 2) * 2 + new_z - z / 2;
#else
                            int linear_idx = idx;
#endif
                            T weight = wt[idx];
                            if (weight == 0) {
                                promat->entryCol[i * pow(kernel_range, dim) + linear_idx] = promat->entryCol[i * pow(kernel_range, dim)];
                                promat->entryVal[i * pow(kernel_range, dim) + linear_idx] = TM::Zero();
                                continue;
                            }
                            ULL hash_key = (ULL)new_x * hash_seed * hash_seed + (ULL)new_y * hash_seed + (ULL)new_z;
                            //IV hash_key{ new_x, new_y, new_z };
                            if (new_coord2id.find(hash_key) == new_coord2id.end()) {
                                new_id2coord.push_back(IV(new_x, new_y, new_z));
                                new_coord2id[hash_key] = new_id2coord.size() - 1;
                            }
                            int j = new_coord2id[hash_key];
                            //ZIRAN_ASSERT(linear_idx >= 0 && linear_idx < 8, "linear offset computation wrong!");
                            promat->entryCol[i * pow(kernel_range, dim) + linear_idx] = j;
                            promat->entryVal[i * pow(kernel_range, dim) + linear_idx] = weight * TM::Identity();
                        }
            }
            id2coord.swap(new_id2coord);
            resmats[level] = std::make_unique<SpMat>();
            resmats[level]->buildTransposeMatrix(*promats[level], id2coord.size());

            SpMat mat;
            mat.buildCoarseMatrix(*sysmats[level], *promats[level]);
            sysmats[level + 1] = std::make_unique<SpMat>();
            sysmats[level + 1]->buildCoarseMatrix(*resmats[level], mat);
            sysmats[level + 1]->buildDiagonal(HOTSettings::Ainv);
            if (HOTSettings::coarseSolver == 5 || HOTSettings::smoother == 5) {
                sysmats[level + 1]->allocateColorIds();
                markColors(sysmats[level + 1]->colorOrder, sysmats[level + 1]->coloredBlockDofs);
            }
            if ((HOTSettings::coarseSolver == 6 && level + 2 == HOTSettings::levelCnt) || (HOTSettings::smoother == 6 && level + 2 < HOTSettings::levelCnt))
                sysmats[level + 1]->estimate2norm();
            if ((HOTSettings::coarseSolver == 7 && level + 2 == HOTSettings::levelCnt))
                sysmats[level + 1]->setupICSolver();
            ZIRAN_ASSERT(!(HOTSettings::smoother == 7 && level + 2 < HOTSettings::levelCnt), "Do not use IC solver as smoother in the multigrid!");

            dofs.emplace_back((int)id2coord.size());
        }

        setup_logic(obj, mass, HOTSettings::smoother, HOTSettings::coarseSolver, HOTSettings::Ainv);
        setup_parameters(obj, HOTSettings::times, HOTSettings::levelscale, 1e-8);
        obj.init(dofs, resmats, sysmats, promats);
        if (HOTSettings::systemBCProject)
            obj.correctResidualProjection = [&dRhs](TVStack& residual) { residual += dRhs; };
        else {
            obj.correctResidualProjection = [](TVStack&) {};
            obj.sysmats[0]->project = project;
        }

        for (int level = 0; level < (int)dofs.size(); ++level)
            printf("%d-th level has %d dofs | ", level, dofs[level]);
    }
}; // namespace ZIRAN

} // namespace ZIRAN

#endif