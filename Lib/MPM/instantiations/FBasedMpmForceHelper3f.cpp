#include "../Force/FBasedMpmForceHelper.cpp"
namespace ZIRAN {
template class FBasedMpmForceHelper<CorotatedIsotropic<float, 3>>;
template class FBasedMpmForceHelper<LinearCorotated<float, 3>>;
} // namespace ZIRAN
