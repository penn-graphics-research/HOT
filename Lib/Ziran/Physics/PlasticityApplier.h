#ifndef PLASTICITY_APPLIER_H
#define PLASTICITY_APPLIER_H

#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/CS/DataStructure/DisjointRanges.h>
#include <Ziran/Math/Geometry/Rotation.h>
#include <Ziran/Math/Linear/Decomposition.h>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>

#include <algorithm>
#include <cmath>

namespace ZIRAN {

class PlasticityApplierBase {
public:
    virtual ~PlasticityApplierBase()
    {
    }

    virtual void applyPlasticity(const DisjointRanges& subrange, DataManager& data_manager) = 0; // parallel
};

template <class TConst, class TPlastic, class TStrain>
class PlasticityApplier : public PlasticityApplierBase {
public:
    AttributeName<TStrain> strain_name;

    PlasticityApplier(const std::string& strain_name_in)
        : strain_name(strain_name_in)
    {
    }

    // This is assumed to be called in a parallel loop that split all particles/elements to subranges.
    void applyPlasticity(const DisjointRanges& subrange, DataManager& data_manager) override
    {
        auto constitutive_model_name = AttributeName<TConst>(TConst::name());
        auto plastic_name = AttributeName<TPlastic>(TPlastic::name());
        if (!data_manager.exist(constitutive_model_name) || !data_manager.exist(plastic_name))
            return;

        DisjointRanges subset(subrange,
            data_manager.commonRanges(constitutive_model_name,
                strain_name,
                plastic_name));
        for (auto it = data_manager.subsetIter(subset, constitutive_model_name, strain_name, plastic_name); it; ++it) {
            auto& c = it.template get<0>();
            auto& s = it.template get<1>();
            auto& p = it.template get<2>();
            p.projectStrain(c, s);
        }
    }
};

template <class T>
class SnowPlasticity {
public:
    T Jp, psi, theta_c, theta_s, min_Jp, max_Jp;

    SnowPlasticity(T psi_in = 10, T theta_c_in = 2e-2, T theta_s_in = 7.5e-3, T min_Jp_in = 0.6, T max_Jp_in = 20);

    // template <class TConst>
    // void projectDiagonalStrain(TConst& c, Vector<T, TConst::dim>& sigma);
    // Snow plasticity
    // strain s is deformation F
    template <class TConst>
    bool projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain);

    template <class TConst>
    void projectStrainDiagonal(TConst& c, Vector<T, TConst::dim>& sigma);

    template <class TConst>
    void computeSigmaPInverse(TConst& c, const Vector<T, TConst::dim>& sigma_e, Vector<T, TConst::dim>& sigma_p_inv);

    static const char* name();
};

template <class T, int dim>
class VonMisesFixedCorotated {
public:
    using TV = Vector<T, dim>;

    T yield_stress;

    VonMisesFixedCorotated()
        : yield_stress(0)
    {
    }

    VonMisesFixedCorotated(const T yield_stress);

    void setParameters(const T yield_stress_in);

    // strain s is deformation F
    template <class TConst>
    bool projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain);

    void write(std::ostream& out) const
    {
        writeEntry(out, yield_stress);
    }

    static VonMisesFixedCorotated<T, dim> read(std::istream& in)
    {
        VonMisesFixedCorotated<T, dim> model;
        model.yield_stress = readEntry<T>(in);
        return model;
    }

    inline static AttributeName<VonMisesFixedCorotated<T, dim>> attributeName()
    {
        return AttributeName<VonMisesFixedCorotated<T, dim>>("VonMisesFixedCorotated");
    }

    static const char* name();
};
} // namespace ZIRAN
#endif
