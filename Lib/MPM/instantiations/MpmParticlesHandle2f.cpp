#include <MPM/MpmParticleHandleBase.cpp>
namespace ZIRAN {
template class MpmParticleHandleBase<float, 2>;
extern template class MpmForceBase<float, 2>;
template void MpmParticleHandleBase<float, 2>::addFBasedMpmForce<CorotatedIsotropic<float, 2>>(CorotatedIsotropic<float, 2> const&);
template void MpmParticleHandleBase<float, 2>::addFBasedMpmForce<LinearCorotated<float, 2>>(LinearCorotated<float, 2> const&);
template void MpmParticleHandleBase<float, 2>::addPlasticity<CorotatedIsotropic<float, 2>, SnowPlasticity<float>>(CorotatedIsotropic<float, 2> const&, SnowPlasticity<float> const&, std::string);
template void MpmParticleHandleBase<float, 2>::addPlasticity<LinearCorotated<float, 2>, SnowPlasticity<float>>(LinearCorotated<float, 2> const&, SnowPlasticity<float> const&, std::string);

} // namespace ZIRAN
