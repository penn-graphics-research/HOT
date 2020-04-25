#include <MPM/MpmParticleHandleBase.cpp>
namespace ZIRAN {

template class MpmParticleHandleBase<float, 3>;

extern template class MpmForceBase<float, 3>;

template void MpmParticleHandleBase<float, 3>::addFBasedMpmForce<CorotatedIsotropic<float, 3>>(CorotatedIsotropic<float, 3> const&);
template void MpmParticleHandleBase<float, 3>::addFBasedMpmForce<LinearCorotated<float, 3>>(LinearCorotated<float, 3> const&);
template void MpmParticleHandleBase<float, 3>::addPlasticity<CorotatedIsotropic<float, 3>, SnowPlasticity<float>>(CorotatedIsotropic<float, 3> const&, SnowPlasticity<float> const&, std::string);
template void MpmParticleHandleBase<float, 3>::addPlasticity<CorotatedIsotropic<float, 3>, VonMisesFixedCorotated<float, 3>>(CorotatedIsotropic<float, 3> const&, VonMisesFixedCorotated<float, 3> const&, std::string);
template void MpmParticleHandleBase<float, 3>::addPlasticity<LinearCorotated<float, 3>, SnowPlasticity<float>>(LinearCorotated<float, 3> const&, SnowPlasticity<float> const&, std::string);
} // namespace ZIRAN
