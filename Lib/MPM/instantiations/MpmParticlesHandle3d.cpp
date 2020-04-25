#include <MPM/MpmParticleHandleBase.cpp>
namespace ZIRAN {
template class MpmParticleHandleBase<double, 3>;
extern template class MpmForceBase<double, 3>;
template void MpmParticleHandleBase<double, 3>::addFBasedMpmForce<CorotatedIsotropic<double, 3>>(CorotatedIsotropic<double, 3> const&);
template void MpmParticleHandleBase<double, 3>::addFBasedMpmForce<LinearCorotated<double, 3>>(LinearCorotated<double, 3> const&);
template void MpmParticleHandleBase<double, 3>::addPlasticity<CorotatedIsotropic<double, 3>, SnowPlasticity<double>>(CorotatedIsotropic<double, 3> const&, SnowPlasticity<double> const&, std::string);
template void MpmParticleHandleBase<double, 3>::addPlasticity<CorotatedIsotropic<double, 3>, VonMisesFixedCorotated<double, 3>>(CorotatedIsotropic<double, 3> const&, VonMisesFixedCorotated<double, 3> const&, std::string);
template void MpmParticleHandleBase<double, 3>::addPlasticity<LinearCorotated<double, 3>, SnowPlasticity<double>>(LinearCorotated<double, 3> const&, SnowPlasticity<double> const&, std::string);
} // namespace ZIRAN
