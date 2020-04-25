#ifndef MULTIGRID_INIT_2D_H
#define MULTIGRID_INIT_2D_H

#include <Ziran/CS/Util/RandomNumber.h>
#include <Ziran/Math/MathTools.h>

#include <Ziran/Sim/MeshHandle.h>
#include <Ziran/Sim/SceneInitializationCore.h>

#include "MultigridSimulation.h"
#include "MultigridInit.h"
#include <float.h>

namespace ZIRAN {

template <class T>
class MultigridInit2D : public MultigridInitBase<T, 2> {
public:
    static const int dim = 2;
    using Base = MultigridInitBase<T, dim>;
    using TV = Vector<T, dim>;
    using TVI = Vector<int, dim>;

    using Base::init_helper;
    using Base::scene;
    using Base::sim;
    using Base::test_number;
    MultigridInit2D(MultigridSimulation<T, dim>& sim, const int test_number)
        : Base(sim, test_number)
    {
    }

    void reload() override
    {
    }
};
} // namespace ZIRAN
#endif
