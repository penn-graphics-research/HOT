#include "../Force/FBasedMpmForceHelper.cpp"
namespace ZIRAN {
template class FBasedMpmForceHelper<CorotatedIsotropic<float, 2>>;
template class FBasedMpmForceHelper<LinearCorotated<float, 2>>;
} // namespace ZIRAN
