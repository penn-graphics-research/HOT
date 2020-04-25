#ifndef DEFORMABLE_OBJECT_HANDLE_CORE_H
#define DEFORMABLE_OBJECT_HANDLE_CORE_H
#include <Ziran/Sim/MeshHandle.h>
#include <Ziran/Math/Geometry/ElementManagerFor.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>
#include <Ziran/Physics/LagrangianForce/LagrangianForce.h>

namespace ZIRAN {

template <class T, int dim, class TMesh>
class DeformableObjectHandle;

template <class T, int dim, int manifold_dim>
class DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>> : public MeshHandle<T, dim, SimplexMesh<manifold_dim>> {
public:
    static const int _manifold_dim = manifold_dim;
    static const int _dim = dim;

    using TMesh = SimplexMesh<manifold_dim>;
    using TElementManager = ElementManagerFor<T, TMesh, dim>;
    using TLagrangianForce = StdVector<std::unique_ptr<LagrangianForce<T, dim>>>;
    using Base = MeshHandle<T, dim, TMesh>;
    using Base::mesh;
    using Base::particle_range;
    using Base::particles;
    TElementManager& elements;
    Range element_range;
    TLagrangianForce& forces;
    DeformableObjectHandle(const Base& undeformed, TElementManager& elements, Range element_range, TLagrangianForce& forces);

    DeformableObjectHandle(const Base& undeformed, TElementManager& elements, TLagrangianForce& forces);

    Base copy();

    T totalVolume() const;

    DeformableObjectHandle subset(Range& subrange);

    void scaleDmInverse(int frame, const std::function<T(int)>& growCurve);

    void resetDmInverse();

    // Add 'no_write' label to the elements in this handle.
    // This is for tracking things like bending springs, where we don't want them to be in 'segmesh_to_write'
    void labelNoWrite();

    void setMassFromDensity(T density)
    {
        elements.setMassFromDensity(density, element_range, particles.mass.array);
    }
};

} // namespace ZIRAN
#endif
