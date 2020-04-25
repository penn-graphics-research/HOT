#ifndef MPM_MULTIGRID_MATRIX_H
#define MPM_MULTIGRID_MATRIX_H

#include "SquareMatrix.h"
#include <tbb/tbb.h>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

namespace ZIRAN {

template <typename T, typename Integer>
constexpr T pow(T val, Integer power)
{
    //if constexpr(power > 0) return val * pow(val, power - 1);
    if (power > 0) return pow(val, power - 1) * val;
    return static_cast<T>(1);
}

template <class T, class IndexType, int dim>
class SparseMPMMatrix { // row-major, quadratic kernel
public:
    static constexpr auto Dim = dim;
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using SpMat = SquareMatrix<T, dim>;
    struct SpIter {
        SpMat& mat;
        int st, ed, r;
        SpIter(SpMat& mat, int row)
            : mat(mat), st(row * mat.colsize), ed(st + mat.colsize), r(row) {}
        auto row() const { return r; }
        auto col() const { return mat.entryCol[st]; }
        auto value() const { return mat.entryVal[st]; }
        bool operator()() { return st < ed; }
        SpIter& operator++()
        {
            st++;
            return *this;
        } ///< prefix
    };
    using Vec = Vector<T, Eigen::Dynamic>;

    std::function<void(TVStack&)> project;
    std::unique_ptr<SpMat> _mat; ///< prolongation or system matrix
    std::unique_ptr<SpMat> _matT; ///< restriction matrix

public:
    SparseMPMMatrix() = default;
    SparseMPMMatrix(std::unique_ptr<SpMat>& sys)
        : _mat(std::move(sys))
    {
        project = [](TVStack&) {};
    }
    SparseMPMMatrix(std::unique_ptr<SpMat>& res, std::unique_ptr<SpMat>& pro)
        : _mat(std::move(pro)), _matT(std::move(res)) {}
    ~SparseMPMMatrix()
    { /*printf("\nDESTROYS SPMAT!\n");*/
    }
    void multiply(const TVStack& x, TVStack& b) const
    {
        _mat->multiply(x, b);
    }
    void transposeMultiply(const TVStack& x, TVStack& b) const
    {
        _matT->multiply(x, b);
    }
};

} // namespace ZIRAN

#endif
