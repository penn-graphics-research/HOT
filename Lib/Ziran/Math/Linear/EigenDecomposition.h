#ifndef EIGEN_DECOMPOSITION_H
#define EIGEN_DECOMPOSITION_H
#include <Ziran/CS/Util/Forward.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Math/Linear/DenseExt.h>

namespace ZIRAN {

/*
template <typename TT, int dim>
inline void fastEigenvalues(const Eigen::Matrix<TT, dim, dim>& A_Sym, Eigen::Matrix<TT, dim, 1>& lambda) // 24 mults, 20 adds, 1 atan2, 1 sincos, 2 sqrts
{
    using T = double;
    using std::max;
    using std::sqrt;
    using std::swap;
    T m = ((T)1 / 3) * (A_Sym(0, 0) + A_Sym(1, 1) + A_Sym(2, 2));
    T a00 = A_Sym(0, 0) - m;
    T a11 = A_Sym(1, 1) - m;
    T a22 = A_Sym(2, 2) - m;
    T a12_sqr = A_Sym(0, 1) * A_Sym(0, 1);
    T a13_sqr = A_Sym(0, 2) * A_Sym(0, 2);
    T a23_sqr = A_Sym(1, 2) * A_Sym(1, 2);
    T p = ((T)1 / 6) * (a00 * a00 + a11 * a11 + a22 * a22 + 2 * (a12_sqr + a13_sqr + a23_sqr));
    T q = (T).5 * (a00 * (a11 * a22 - a23_sqr) - a11 * a13_sqr - a22 * a12_sqr) + A_Sym(0, 1) * A_Sym(0, 2) * A_Sym(1, 2);
    T sqrt_p = sqrt(p);
    T disc = p * p * p - q * q;
    T phi = ((T)1 / 3) * atan2(sqrt(max((T)0, disc)), q);
    T c = cos(phi), s = sin(phi);
    T sqrt_p_cos = sqrt_p * c;
    T root_three_sqrt_p_sin = sqrt((T)3) * sqrt_p * s;
    lambda(0) = m + 2 * sqrt_p_cos;
    lambda(1) = m - sqrt_p_cos - root_three_sqrt_p_sin;
    lambda(2) = m - sqrt_p_cos + root_three_sqrt_p_sin;
    if (lambda(0) < lambda(1))
        swap(lambda(0), lambda(1));
    if (lambda(1) < lambda(2))
        swap(lambda(1), lambda(2));
    if (lambda(0) < lambda(1))
        swap(lambda(0), lambda(1));
}
//#####################################################################
// Function Fast_Eigenvectors
//#####################################################################
template <typename TT, int dim>
inline void fastEigenvectors(const Eigen::Matrix<TT, dim, dim>& A_Sym,
    const Eigen::Matrix<TT, dim, 1>& lambda, Eigen::Matrix<TT, dim, dim>& V) // 71 mults, 44 adds, 3 divs, 3 sqrts
{
    // flip if necessary so that first eigenvalue is the most different
    using T = double;
    using std::sqrt;
    using std::swap;
    using ZIRAN::EIGEN_EXT::cofactorMatrix;
    bool flipped = false;
    Eigen::Vector3d lambda_flip(lambda);
    if (lambda(0) - lambda(1) < lambda(1) - lambda(2)) { // 2a
        swap(lambda_flip(0), lambda_flip(2));
        flipped = true;
    }

    // get first eigenvector
    Eigen::Matrix3d C1;
    EIGEN_EXT::cofactorMatrix(A_Sym - lambda_flip(0) * Eigen::Matrix3d::Identity(), C1);
    Eigen::Matrix3d::Index i;
    T norm2 = C1.colwise().squaredNorm().maxCoeff(&i); // 3a + 12m+6a + 9m+6a+1d+1s = 21m+15a+1d+1s
    Eigen::Vector3d v1;
    if (norm2 != 0) {
        T one_over_sqrt = (T)1 / sqrt(norm2);
        v1 = C1.col(i) * one_over_sqrt;
    }
    else
        v1 << 1, 0, 0;

    // form basis for orthogonal complement to v1, and reduce A to this space
    Eigen::Vector3d v1_orthogonal = v1.unitOrthogonal(); // 6m+2a+1d+1s (tweak: 5m+1a+1d+1s)
    Eigen::Matrix<T, 3, 2> other_v;
    other_v.col(0) = v1_orthogonal;
    other_v.col(1) = v1.cross(v1_orthogonal); // 6m+3a (tweak: 4m+1a)
    Eigen::Matrix2d A_reduced = other_v.transpose() * A_Sym * other_v; // 21m+12a (tweak: 18m+9a)

    // find third eigenvector from A_reduced, and fill in second via cross product
    Eigen::Matrix2d C3;
    EIGEN_EXT::cofactorMatrix(A_reduced - lambda_flip(2) * Eigen::Matrix2d::Identity(), C3);
    Eigen::Matrix2d::Index j;
    norm2 = C3.colwise().squaredNorm().maxCoeff(&j); // 3a + 12m+6a + 9m+6a+1d+1s = 21m+15a+1d+1s
    Eigen::Vector3d v3;
    if (norm2 != 0) {
        T one_over_sqrt = (T)1 / sqrt(norm2);
        v3 = other_v * C3.col(j) * one_over_sqrt;
    }
    else
        v3 = other_v.col(0);

    Eigen::Vector3d v2 = v3.cross(v1); // 6m+3a

    // finish
    if (flipped) {
        V.col(0) = v3;
        V.col(1) = v2;
        V.col(2) = -v1;
    }
    else {
        V.col(0) = v1;
        V.col(1) = v2;
        V.col(2) = v3;
    }
}
template <typename T, int dim>
void makePD(Matrix<T, dim, dim>& mat)
{
    Eigen::Matrix<T, dim, 1> lambda;
    Eigen::Matrix<T, dim, dim> V;
    fastEigenvalues(mat, lambda);
    fastEigenvectors(mat, lambda, V);
    for (int i = 0; i < dim; ++i)
        lambda[i] = lambda[i] > 1e-6 ? lambda[i] : 1e-6; ///< clamp
    //matrixTranspose(vecs, VT);
    //matrixDiagonalMatrixMultiplication(vecs, vals, tmp);
    //matrixMatrixMultiplication(tmp, VT, mat);
    mat = V * lambda.asDiagonal() * V.transpose();
}
*/

template <class T, int dim>
void makePD(Eigen::Matrix<T, dim, dim>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigenSolver(symMtr);
    Eigen::Matrix<T, dim, 1> D(eigenSolver.eigenvalues());
    for (int i = 0; i < dim; i++)
        if (D[i] < 0.0)
            D[i] = 0.0;
    symMtr = eigenSolver.eigenvectors() * D.asDiagonal() * eigenSolver.eigenvectors().transpose();
};

} // namespace ZIRAN

#endif