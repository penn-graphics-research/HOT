#include "../Force/FBasedMpmForceHelper.cpp"
namespace ZIRAN {
template class FBasedMpmForceHelper<CorotatedIsotropic<double, 2>>;
template class FBasedMpmForceHelper<LinearCorotated<double, 2>>;
} // namespace ZIRAN
