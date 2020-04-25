#pragma once

#include "MpmParticleHandleBase.h"
#include <MPM/MpmSimulationBase.h>

#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/CS/DataStructure/KdTree.h>
#include <Ziran/CS/Util/DataDir.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Ziran/Math/Geometry/ObjIO.h>
#include <Ziran/Math/Geometry/PoissonDisk.h>
#include <Ziran/Math/Geometry/VoronoiNoise.h>
#include <Ziran/Math/Geometry/AnalyticLevelSet.h>
#include <Ziran/Math/Geometry/VdbLevelSet.h>
#include <Ziran/Math/Geometry/VtkIO.h>
#include <Ziran/Physics/LagrangianForce/LagrangianForce.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>
#include <Ziran/Physics/PlasticityApplier.h>
#include <Ziran/Sim/Scene.h>

#include <MPM/Force/FBasedMpmForceHelper.h>
#include <MPM/Force/LinearCorotatedMpmForceHelper.h>
#include <MPM/Force/JBasedMpmForceHelper.h>

namespace ZIRAN {

template <class T, int dim>
MpmParticleHandleBase<T, dim>::
    MpmParticleHandleBase(Particles<T, dim>& particles, Scene<T, dim>& scene, MpmForceBase<T, dim>* mpmforce,
        StdVector<std::unique_ptr<PlasticityApplierBase>>& plasticity_appliers,
        StdVector<TV>& scratch_xp, T& dt, Range particle_range, T total_volume, int cotangent_manifold_dim)
    : particles(particles)
    , scene(scene)
    , mpmforce(mpmforce)
    , plasticity_appliers(plasticity_appliers)
    , scratch_xp(scratch_xp)
    , dt(dt)
    , particle_range(particle_range)
    , total_volume(total_volume)
    , cotangent_manifold_dim(cotangent_manifold_dim)
{
}

// Creates a copy with new particles
template <class T, int dim>
MpmParticleHandleBase<T, dim>
MpmParticleHandleBase<T, dim>::copy()
{
    Range new_particle_range;
    new_particle_range.lower = particles.count;
    {
        auto ap = particles.appender();
        for (int i = particle_range.lower; i < particle_range.upper; i++)
            ap.append(particles.mass[i], particles.X[i], particles.V[i]);
    }
    new_particle_range.upper = particles.count;
    return MpmParticleHandleBase(particles, scene, mpmforce, plasticity_appliers, scratch_xp, dt, new_particle_range, total_volume);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::transform(const std::function<void(int, Ref<T>, Vector<T, dim>&, Vector<T, dim>&)>& mapping)
{
    for (int i = particle_range.lower; i < particle_range.upper; ++i) {
        // lua does not support passing scalars by reference. This is a work around to actually change mass.
        mapping(i - particle_range.lower, particles.mass[i], particles.X[i], particles.V[i]);
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addVolumeFraction(const T b)
{
    particles.add(volume_fraction_name<T>(), particle_range, b);
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::
    addFBasedMpmForce(const TCONST& model)
{
    if (cotangent_manifold_dim == 0)
        addFBasedMpmForceWithMeasure(model, particle_range, total_volume);
    else
        ZIRAN_ASSERT(false);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::addOriginalPositionAsAttribute()
{
    // Create X0 as attribute
    particles.add(X0_name<T, dim>(), particle_range, TV::Zero());

    // Fill in X0 with current X
    for (auto iter = particles.iter(X0_name<T, dim>()); iter; ++iter) {
        Vector<T, dim>& X0 = iter.template get<0>();
        X0 = particles.X(iter.entryId());
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    scaleF(const T scale)
{
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        F *= scale;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    scaleF(const TV scale)
{
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        if constexpr (dim >= 1)
            F.row(0) *= scale[0];
        if constexpr (dim >= 2)
            F.row(1) *= scale[1];
        if constexpr (dim >= 3)
            F.row(2) *= scale[2];
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    scaleF(const std::vector<TV>& scales)
{
    int particleI = 0;
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
        if (particleI >= (int)scales.size()) {
            break;
        }

        auto& F = iter.template get<0>();
        if constexpr (dim >= 1)
            F.row(0) *= scales[particleI][0];
        if constexpr (dim >= 2)
            F.row(1) *= scales[particleI][1];
        if constexpr (dim >= 3)
            F.row(2) *= scales[particleI][2];

        ++particleI;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    scaleJ(const T scale)
{
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, J_name<T>()); iter; ++iter) {
        auto& J = iter.template get<0>();
        J *= scale;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    scaleFCurve(int frame, const std::function<T(int)>& growCurve)
{
    DisjointRanges subset{ particle_range };
    for (auto iter = particles.subsetIter(subset, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        if (frame == 0)
            F /= growCurve(0);
        // else
        //      F *= (growCurve(frame - 1) / growCurve(frame));
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    resetDeformation()
{
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        F = Matrix<T, dim, dim>::Identity();
    }
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::
    addJBasedMpmForce(const TCONST& model)
{
    JBasedMpmForceHelper<TCONST>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    particles.add(helper.constitutive_model_name(), particle_range, model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
        particles.add(J_name<T>(), particle_range, (T)1);
    }
    particles.add(helper.constitutive_model_scratch_name(), particle_range, typename JBasedMpmForceHelper<TCONST>::Scratch());
    //addJBasedMpmForceWithMeasure(model, particle_range, total_volume);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addLinearCorotatedMpmForce(const LinearCorotated<T, dim>& model)
{
    LinearCorotatedMpmForceHelper<T, dim>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    particles.add(helper.constitutive_model_name(), particle_range, model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
        particles.add(helper.F_name(), particle_range, TM::Identity());
    }
    particles.add(helper.constitutive_model_scratch_name(), particle_range, typename LinearCorotatedMpmForceHelper<T, dim>::Scratch());
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addElementMeasure()
{
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
    }
}

template <class T, int dim>
template <class TConst, class TPlastic>
void MpmParticleHandleBase<T, dim>::
    addPlasticity(const TConst& cons, const TPlastic& plasticity, std::string strain_name)
{
    using TStrain = typename TConst::Strain;
    PlasticityApplier<TConst, TPlastic, TStrain>* plasticity_model = nullptr;
    for (auto& p : plasticity_appliers) {
        plasticity_model = dynamic_cast<PlasticityApplier<TConst, TPlastic, TStrain>*>(p.get());
        if (plasticity_model && plasticity_model->strain_name.name == strain_name)
            break;
        else
            plasticity_model = nullptr;
    }
    if (plasticity_model == nullptr)
        plasticity_appliers.push_back(std::make_unique<PlasticityApplier<TConst, TPlastic, TStrain>>(strain_name));
    particles.add(TPlastic::name(), particle_range, plasticity);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    prescoreSnow(AnalyticLevelSet<T, dim>& levelset, T grain_size, T density_min, T Jp_min)
{
    ZIRAN_INFO("Prescoring Snow Particles");
    TV min_corner, max_corner;
    levelset.getBounds(min_corner, max_corner);
    PoissonDisk<T, dim> pd(/*random seed*/ 123, grain_size, min_corner, max_corner);
    StdVector<TV> grain_centroids;
    if (dim == 3)
        pd.sampleFromPeriodicData(grain_centroids, [&](const TV& x) { return levelset.inside(x); });
    else
        pd.sample(grain_centroids, [&](const TV& x) { return levelset.inside(x); });
    auto spn = AttributeName<SnowPlasticity<T>>(SnowPlasticity<T>::name());
    T phi = T(0.5) * (1 - std::sqrt((T)5));
    RandomNumber<T> rand;
    for (auto iter = particles.subsetIter({ particle_range }, Particles<T, dim>::X_name(), Particles<T, dim>::mass_name(), spn, element_measure_name<T>()); iter; ++iter) {
        TV& x = iter.template get<0>();
        for (const TV& c : grain_centroids) {
            TV v = c - x;
            T alpha = std::min(v.norm() / ((T)1.2 * grain_size), (T)1);
            T scale = phi + 1 / (alpha - phi);
            scale *= rand.randReal(0.75, 1.0);
            x = x + scale * v;
        }
        T min_distance2 = std::numeric_limits<T>::max();
        for (const TV& c : grain_centroids)
            min_distance2 = std::min(min_distance2, (x - c).squaredNorm());
        T& mass = iter.template get<1>();
        SnowPlasticity<T>& p = iter.template get<2>();
        T element_measure = iter.template get<3>();
        T scale = 1 - std::min(std::sqrt(min_distance2) * rand.randReal(0.75, 1.25) / (2 * grain_size), (T)1);
        T mass_min = density_min * element_measure;
        mass = scale * (mass - mass_min) + mass_min;
        p.theta_c *= scale;
        p.theta_s *= scale;
        p.Jp = scale * (p.max_Jp - Jp_min) + Jp_min;
    }
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::
    addFBasedMpmForceWithMeasure(const TCONST& model, const Range& range, T total_volume)
{
    FBasedMpmForceHelper<TCONST>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    particles.add(helper.constitutive_model_name(), range, model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), range, total_volume / range.length());
        particles.add(helper.F_name(), range, TM::Identity());
    }
    particles.add(helper.constitutive_model_scratch_name(), range, typename FBasedMpmForceHelper<TCONST>::Scratch());
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::setMassFromDensity(const T density)
{
    ZIRAN_INFO("Setting mass from densiy: total_volume = ", total_volume, ", particle.count = ", particle_range.length());
    T mp = density * total_volume / particle_range.length();
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, particles.mass_name()); iter; ++iter) {
        iter.template get<0>() = mp;
    }
}
} // namespace ZIRAN
