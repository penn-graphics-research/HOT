#include "PlasticityApplier.h"

namespace ZIRAN {

template <class T>
SnowPlasticity<T>::SnowPlasticity(T psi_in, T theta_c_in, T theta_s_in, T min_Jp_in, T max_Jp_in)
    : Jp(1)
    , psi(psi_in)
    , theta_c(theta_c_in)
    , theta_s(theta_s_in)
    , min_Jp(min_Jp_in)
    , max_Jp(max_Jp_in)
{
}

template <class T>
template <class TConst>
bool SnowPlasticity<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    T Fe_det = (T)1;
    for (int i = 0; i < dim; i++) {
        sigma(i) = std::max(std::min(sigma(i), (T)1 + theta_s), (T)1 - theta_c);
        Fe_det *= sigma(i);
    }

    Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
    TM Fe = U * sigma_m * V.transpose();
    // T Jp_new = std::max(std::min(Jp * strain.determinant() / Fe_det, max_Jp), min_Jp);
    T Jp_new = Jp * strain.determinant() / Fe_det;
    if (!(Jp_new <= max_Jp))
        Jp_new = max_Jp;
    if (!(Jp_new >= min_Jp))
        Jp_new = min_Jp;

    strain = Fe;
    c.mu *= std::exp(psi * (Jp - Jp_new));
    c.lambda *= std::exp(psi * (Jp - Jp_new));
    Jp = Jp_new;

    return false;
}

template <class T>
template <class TConst>
void SnowPlasticity<T>::projectStrainDiagonal(TConst& c, Vector<T, TConst::dim>& sigma)
{
    static const int dim = TConst::dim;

    for (int i = 0; i < dim; i++) {
        sigma(i) = std::max(std::min(sigma(i), (T)1 + theta_s), (T)1 - theta_c);
    }
}

template <class T>
template <class TConst>
void SnowPlasticity<T>::computeSigmaPInverse(TConst& c, const Vector<T, TConst::dim>& sigma_e, Vector<T, TConst::dim>& sigma_p_inv)
{
    using TV = typename TConst::TV;
    TV sigma_proj = sigma_e;
    projectStrainDiagonal(c, sigma_proj);
    sigma_p_inv.array() = sigma_proj.array() / sigma_e.array();

    ZIRAN_WARN("Snow lambda step not fully implemented yet. ");
}

template <class T>
const char* SnowPlasticity<T>::name()
{
    return "SnowPlasticity";
}

template <class T, int dim>
VonMisesFixedCorotated<T, dim>::VonMisesFixedCorotated(const T yield_stress)
    : yield_stress(yield_stress)
{
}

template <class T, int dim>
void VonMisesFixedCorotated<T, dim>::setParameters(const T yield_stress_in)
{
    yield_stress = yield_stress_in;
}

// strain s is deformation F
template <class T, int dim>
template <class TConst>
bool VonMisesFixedCorotated<T, dim>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static_assert(dim == TConst::dim, "Plasticity model has a different dimension as the Constitutive model!");
    ZIRAN_ASSERT(yield_stress >= 0);
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    using MATH_TOOLS::sqr;

    TM U, V;
    TV sigma;
    singularValueDecomposition(strain, U, sigma, V);
    for (int d = 0; d < dim; d++) sigma(d) = std::max((T)1e-4, sigma(d));
    T J = sigma.prod();
    TV tau_trial;
    for (int d = 0; d < dim; d++) tau_trial(d) = 2 * c.mu * (sigma(d) - 1) * sigma(d) + c.lambda * (J - 1) * J;
    T trace_tau = tau_trial.sum();
    TV s_trial = tau_trial - TV::Ones() * (trace_tau / (T)dim);
    T s_norm = s_trial.norm();
    T scaled_tauy = std::sqrt((T)2 / ((T)6 - dim)) * yield_stress;
    if (s_norm - scaled_tauy <= 0) return false;
    T alpha = scaled_tauy / s_norm;
    TV s_new = alpha * s_trial;
    TV tau_new = s_new + TV::Ones() * (trace_tau / (T)dim);
    TV sigma_new;
    for (int d = 0; d < dim; d++) {
        T b2m4ac = sqr(c.mu) - 2 * c.mu * (c.lambda * (J - 1) * J - tau_new(d));
        ZIRAN_ASSERT(b2m4ac >= 0, "Wrong projection ", b2m4ac);
        T sqrtb2m4ac = std::sqrt(b2m4ac);
        T x1 = (c.mu + sqrtb2m4ac) / (2 * c.mu);
        // T x2 = (c.mu - sqrtb2m4ac) / (2 * c.mu);
        // ZIRAN_ASSERT(sqr(x1 - sigma(d)) <= sqr(x2 - sigma(d)));
        sigma_new(d) = x1;
    }
    strain = U * sigma_new.asDiagonal() * V.transpose();
    return true;
}

template <class T, int dim>
const char* VonMisesFixedCorotated<T, dim>::name()
{
    return "VonMisesFixedCorotated";
}

template class SnowPlasticity<double>;
template class SnowPlasticity<float>;

template class VonMisesFixedCorotated<double, 2>;
template class VonMisesFixedCorotated<float, 2>;
template class VonMisesFixedCorotated<double, 3>;
template class VonMisesFixedCorotated<float, 3>;

template class PlasticityApplier<CorotatedIsotropic<double, 2>, SnowPlasticity<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<CorotatedIsotropic<double, 3>, SnowPlasticity<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<CorotatedIsotropic<float, 2>, SnowPlasticity<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<CorotatedIsotropic<float, 3>, SnowPlasticity<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<CorotatedIsotropic<double, 2>, VonMisesFixedCorotated<double, 2>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<CorotatedIsotropic<double, 3>, VonMisesFixedCorotated<double, 3>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<CorotatedIsotropic<float, 2>, VonMisesFixedCorotated<float, 2>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<CorotatedIsotropic<float, 3>, VonMisesFixedCorotated<float, 3>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template class PlasticityApplier<LinearCorotated<double, 2>, SnowPlasticity<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<LinearCorotated<double, 3>, SnowPlasticity<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<LinearCorotated<float, 2>, SnowPlasticity<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<LinearCorotated<float, 3>, SnowPlasticity<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template void SnowPlasticity<double>::projectStrainDiagonal<CorotatedIsotropic<double, 2>>(CorotatedIsotropic<double, 2>&, Eigen::Matrix<double, CorotatedIsotropic<double, 2>::dim, 1, 0, CorotatedIsotropic<double, 2>::dim, 1>&);
template void SnowPlasticity<double>::projectStrainDiagonal<CorotatedIsotropic<double, 3>>(CorotatedIsotropic<double, 3>&, Eigen::Matrix<double, CorotatedIsotropic<double, 3>::dim, 1, 0, CorotatedIsotropic<double, 3>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<CorotatedIsotropic<float, 2>>(CorotatedIsotropic<float, 2>&, Eigen::Matrix<float, CorotatedIsotropic<float, 2>::dim, 1, 0, CorotatedIsotropic<float, 2>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<CorotatedIsotropic<float, 3>>(CorotatedIsotropic<float, 3>&, Eigen::Matrix<float, CorotatedIsotropic<float, 3>::dim, 1, 0, CorotatedIsotropic<float, 3>::dim, 1>&);

template bool SnowPlasticity<double>::projectStrain<CorotatedIsotropic<double, 2>>(CorotatedIsotropic<double, 2>&, Eigen::Matrix<double, CorotatedIsotropic<double, 2>::dim, CorotatedIsotropic<double, 2>::dim, 0, CorotatedIsotropic<double, 2>::dim, CorotatedIsotropic<double, 2>::dim>&);
template bool SnowPlasticity<double>::projectStrain<LinearCorotated<double, 2>>(LinearCorotated<double, 2>&, Eigen::Matrix<double, LinearCorotated<double, 2>::dim, LinearCorotated<double, 2>::dim, 0, LinearCorotated<double, 2>::dim, LinearCorotated<double, 2>::dim>&);
template bool SnowPlasticity<float>::projectStrain<CorotatedIsotropic<float, 2>>(CorotatedIsotropic<float, 2>&, Eigen::Matrix<float, CorotatedIsotropic<float, 2>::dim, CorotatedIsotropic<float, 2>::dim, 0, CorotatedIsotropic<float, 2>::dim, CorotatedIsotropic<float, 2>::dim>&);
template bool SnowPlasticity<float>::projectStrain<LinearCorotated<float, 2>>(LinearCorotated<float, 2>&, Eigen::Matrix<float, LinearCorotated<float, 2>::dim, LinearCorotated<float, 2>::dim, 0, LinearCorotated<float, 2>::dim, LinearCorotated<float, 2>::dim>&);

} // namespace ZIRAN
