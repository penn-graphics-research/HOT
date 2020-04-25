#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/KrylovSolvers.h>
#include <Ziran/Math/Linear/Minres.h>
#include <Ziran/Math/Linear/GeneralizedMinimalResidual.h>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Sim/SimulationBase.h>
#include <Ziran/Math/Linear/EigenSparseLU.h>

#include <tbb/tbb.h>
#include "MultigridPreconditioner.h"
#include "function_ref.hpp"

namespace ZIRAN {

template <class T, int dim>
class SparseMatrix {
public:
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using Vec = Vector<T, Eigen::Dynamic>;

    using Scalar = T;
    using NewtonVector = TVStack;

    std::vector<int>& entryCol;
    std::vector<TM>& entryVal;
    std::vector<IV>& id2coord;

    std::function<void(const TVStack&, TVStack&)>& precondition;

    SparseMatrix(std::vector<int>& entryCol, std::vector<TM>& entryVal, std::vector<IV>& id2coord, std::function<void(const TVStack&, TVStack&)>& precondition)
        : entryCol(entryCol)
        , entryVal(entryVal)
        , id2coord(id2coord)
        , precondition(precondition)
    {
    }

    ~SparseMatrix() {}

    auto& rebuildPreconditioner(Vec& mass, std::function<void(TVStack&)>& project, const TVStack& dRhs, int numLevel, int times, T omega, const Vec& nodeCNTol, bool check = false)
    {
        MultigridBuilder<T, int, dim> mgbuilder;
        static MultigridOperator<T, int, dim> mgp;
        if (check) {
            //mgp.checkSystemMatrix();
            return mgp;
        }
        if (!id2coord.size()) return mgp;
        mgbuilder.build(mgp, mass, project, dRhs, numLevel, id2coord, entryCol, entryVal, nodeCNTol);
        precondition = tl::function_ref<void(const TVStack&, TVStack&)>(mgp); //tl::function_ref<MultigridPreconditioner<T, int, dim>>(mgp);
        return mgp;
    }

    void multiply(const TVStack& x, TVStack& b) const
    {
        int row_cnt = x.cols();
        int colsize = pow(5, dim);
        tbb::parallel_for(0, (int)row_cnt, [&](int i) {
            TV sum = TV::Zero();
            int idx = i * colsize;
            for (int j = 0; j < colsize; ++idx, ++j) {
                int col_idx = entryCol[idx];
                sum += entryVal[idx] * x.col(col_idx);
            }
            b.col(i) = sum;
        });
    }
};
} // namespace ZIRAN

#endif
