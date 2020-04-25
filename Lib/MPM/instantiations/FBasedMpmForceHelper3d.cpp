#include "../Force/FBasedMpmForceHelper.cpp"
namespace ZIRAN {
template class FBasedMpmForceHelper<CorotatedIsotropic<double, 3>>;
template class FBasedMpmForceHelper<LinearCorotated<double, 3>>;
} // namespace ZIRAN
