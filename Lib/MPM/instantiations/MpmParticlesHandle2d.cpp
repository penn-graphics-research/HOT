#include <MPM/MpmParticleHandleBase.cpp>
namespace ZIRAN {
template class MpmParticleHandleBase<double, 2>;
extern template class MpmForceBase<double, 2>;
template void MpmParticleHandleBase<double, 2>::addFBasedMpmForce<CorotatedIsotropic<double, 2>>(CorotatedIsotropic<double, 2> const&);
template void MpmParticleHandleBase<double, 2>::addFBasedMpmForce<LinearCorotated<double, 2>>(LinearCorotated<double, 2> const&);
template void MpmParticleHandleBase<double, 2>::addPlasticity<CorotatedIsotropic<double, 2>, SnowPlasticity<double>>(CorotatedIsotropic<double, 2> const&, SnowPlasticity<double> const&, std::string);
template void MpmParticleHandleBase<double, 2>::addPlasticity<LinearCorotated<double, 2>, SnowPlasticity<double>>(LinearCorotated<double, 2> const&, SnowPlasticity<double> const&, std::string);
} // namespace ZIRAN
