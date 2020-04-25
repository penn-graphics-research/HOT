#ifndef MULTIGRID_INIT_3D_H
#define MULTIGRID_INIT_3D_H

#include <Ziran/CS/Util/RandomNumber.h>
#include <Ziran/Math/Geometry/MeshConstruction.h>
#include <Ziran/Physics/SoundSpeedCfl.h>
#include <Ziran/Sim/MeshHandle.h>
#include <Ziran/Sim/SceneInitializationCore.h>
#include "MultigridSimulation.h"
#include "MultigridInit.h"
#include "Configurations.h"

namespace ZIRAN {

template <class T, int dim>
class MultigridInitBase;

template <class T>
class MultigridInit3D : public MultigridInitBase<T, 3> {
public:
    static const int dim = 3;
    using Base = MultigridInitBase<T, dim>;
    using TV2 = Vector<T, 2>;
    using TVI2 = Vector<int, 2>;
    using TV = Vector<T, dim>;
    using TM = Eigen::Matrix<T, dim, dim>;
    using TVI = Vector<int, dim>;

    using Base::init_helper;
    using Base::scene;
    using Base::sim;
    using Base::test_number;

    MultigridInit3D(MultigridSimulation<T, dim>& sim, const int test_number)
        : Base(sim, test_number)
    {
    }

    void reload() override
    {
        /// case name, dt, dx, flags, young's modulus, ppc, level, times, levelscale, omega, fashion, sm0, sm1, scaler
        auto destination = [](std::string tag, double dt, double dx, double E, int ppc, int np) {
            auto solverStrs = [](int opt) -> const char* {
                switch (opt) {
                case 0: return "MGS";
                case 1: return "Minres";
                case 2: return "PCG";
                case 3: return "LBFGS";
                default: return "Err";
                }
            };
            auto smootherStrs = [](int opt) -> const char* {
                switch (opt) {
                case 0: return "J";
                case 1: return "OJ";
                case 2: return "PCG";
                case 3: return "LBFGS(disabled)";
                case 4: return "Minres";
                case 5: return "StrictGS";
                case 6: return "ApproxGS";
                default: return "Err";
                }
            };
            char filename[256];
            int solver_flags = (HOTSettings::linesearch << 4) | (HOTSettings::useCN << 3) | (HOTSettings::project << 2) | (HOTSettings::systemBCProject << 1) | HOTSettings::matrixFree;
            sprintf(filename, "%s-%s-dt%.2e-dx%.2e-(%x=ls-cn-SPD-SysBCProj-MatFree)-E%.1e-ppc%d-l%d-t%d-s%d-o%.2e-%s-%s-%s-%s-p%d_cneps%.1e", tag.data(), solverStrs(HOTSettings::lsolver), dt, dx, solver_flags, E, ppc, HOTSettings::levelCnt, HOTSettings::times, HOTSettings::levelscale, HOTSettings::omega, !HOTSettings::topDownMGS ? "vcyle" : "topdown", smootherStrs(HOTSettings::smoother), smootherStrs(HOTSettings::coarseSolver), HOTSettings::Ainv == 0 ? "Entry" : (HOTSettings::Ainv == 1 ? "Block" : "Mass"), np, HOTSettings::cneps);
            return std::string(filename);
        };

        if (test_number == 9211) {
            sim.end_frame = 60;
            sim.dx = 4e-2;
            sim.gravity = -9.8 * TV::Unit(1);
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;

            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            //sim.quasistatic = false;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.write_substeps = false;
            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt;
            //sim.step.max_dt = 2e-3;

            T density[2] = { 2500, 1000 };
            T E[2] = { 100e9, 2.5e4 }, nu[2] = { 0.4, 0.4 };

            T cubeELen = 1.0;
            TV cubeOrigin(10.0, sim.dx * 10, 10.0);
            int ppc = 8;

            AxisAlignedAnalyticBox<T, dim> cube(cubeOrigin + TV(cubeELen * 0.2, 0.0, cubeELen * 0.2), cubeOrigin + TV(cubeELen * 1.8, cubeELen, cubeELen * 1.8)); //rectangular level set
            MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(cube, density[1], ppc);
            CorotatedIsotropic<T, dim> model(E[1], nu[1]);
            model.project = HOTSettings::project;
            particles_handle.addFBasedMpmForce(model);

            CorotatedIsotropic<T, dim> model2(E[0], nu[0]);
            MpmParticleHandleBase<T, dim> particles_handle_top = init_helper.sampleFromVdbFile(
                "LevelSets/luckyCat_rot.vdb",
                density[0], ppc);
            auto initial_translation_kb = [&](int index, Ref<T> mass, TV& X, TV& V) {
                X = X + cubeOrigin + TV(cubeELen * 1.3, cubeELen * 1.6, cubeELen * 0.9);
                V = TV(0.0, -1.5, 0.0);
            };
            particles_handle_top.transform(initial_translation_kb);
            model2.project = HOTSettings::project;
            particles_handle_top.addFBasedMpmForce(model2);

            TV ground_origin(0, sim.dx * 10, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            //AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SLIP);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object); // add ground

            sim.output_dir.path = destination("output/LKCatBounce", sim.step.max_dt, sim.dx, E[1], ppc, sim.particles.count);
        }

        // ./multigrid -test 9212 --3d --usecn -cneps 1e-7 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 2 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 --l2norm -cmd0 1e9 -o LBFGS_1e9_
        // ./multigrid -test 9212 --3d --usecn -cneps 1e-7 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 --l2norm -cmd0 1e9 -o PN_1e9_
        // ./multigrid -test 9212 --3d --usecn -cneps 1e-7 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 --l2norm -cmd0 1e9 -o PN_1e9_MF --matfree
        // ./multigrid -test 9212 --3d --usecn -cneps 1e-7 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 2 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 --l2norm -cmd0 1e6 -o LBFGS_1e6_
        // ./multigrid -test 9212 --3d --usecn -cneps 1e-7 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 --l2norm -cmd0 1e6 -o PN_1e6_
        // ./multigrid -test 9212 --3d --usecn -cneps 1e-7 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 --l2norm -cmd0 1e6 -o PN_1e6_MF --matfree
        if (test_number == 9212) {
            sim.end_frame = 100;
            sim.dx = 4e-2;
            sim.gravity = -9.8 * TV::Unit(1);
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;

            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            //sim.quasistatic = false;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.write_substeps = false;
            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt;

            T E[3] = { CmdArgument::cmd0, 1.0e6, 1.0e5 }, nu[3] = { 0.4, 0.47, 0.4 }, density[3] = { 2500, 1000, 1000 };
            int ppc[3] = { 8, 16, 8 };
            CorotatedIsotropic<T, dim> model2(E[0], nu[0]);
            // MpmParticleHandleBase<T, dim> particles_handle_top = init_helper.sampleFromVdbFile(
            //     "LevelSets/luckyCat_rot.vdb",
            //     density[0], ppc[0]);
            StdVector<TV> meshed_points0;
            readPositionObj("luckyCat20K.obj", meshed_points0);
            MpmParticleHandleBase<T, dim> particles_handle_top = init_helper.sampleFromVdbFileWithExistingPoints(
                meshed_points0, "LevelSets/luckyCat_rot.vdb", density[0], ppc[0]);
            auto initial_translation_kb = [&](int index, Ref<T> mass, TV& X, TV& V) {
                X = X + TV(10.0, 11.3, 10.0);
                V = TV(0.0, -1.5, 0.0);
            };
            particles_handle_top.transform(initial_translation_kb);
            model2.project = HOTSettings::project;
            particles_handle_top.addFBasedMpmForce(model2);

            CorotatedIsotropic<T, dim> model3(E[2], nu[2]);
            // MpmParticleHandleBase<T, dim> particles_handle_arma = init_helper.sampleFromVdbFile(
            //     "LevelSets/Armadillo.vdb",
            //     density[2], ppc[2]);
            StdVector<TV> meshed_points2;
            readPositionObj("Armadillo.obj", meshed_points2);
            MpmParticleHandleBase<T, dim> particles_handle_arma = init_helper.sampleFromVdbFileWithExistingPoints(
                meshed_points2, "LevelSets/Armadillo.vdb", density[2], ppc[2]);
            auto initial_translation_arma = [&](int index, Ref<T> mass, TV& X, TV& V) {
                X = X + TV(10.0, 10.45, 10.6);
                V = TV(0.0, 0.0, 0.0);
            };
            particles_handle_arma.transform(initial_translation_arma);
            model3.project = HOTSettings::project;
            particles_handle_arma.addFBasedMpmForce(model3);

            TV trans(10.0, 10.0, 10.0);
            T theta = (T)0.0 / 180.0 * M_PI;
            Vector<T, 4> rotation(std::cos(theta / 2), 0, 0, std::sin(theta / 2));

            T radius = 2.4, height = 0.08;
            CappedCylinder<T, dim> cylinder(radius, height, rotation, trans);
            // MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(cylinder, density[1], ppc[1]);
            StdVector<TV> meshed_points1;
            readPositionObj("cylinder.obj", meshed_points1);
            for (unsigned int i = 0; i < meshed_points1.size(); ++i) {
                meshed_points1[i] += trans;
            }
            MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSetWithExistingPoints(
                meshed_points1, cylinder, density[1], ppc[1]);
            CorotatedIsotropic<T, dim> model(E[1], nu[1]);
            model.project = HOTSettings::project;
            particles_handle.addFBasedMpmForce(model);

            T rbig = 2.36, rsmall = 0.04;
            Torus<T, dim> torus(rbig, rsmall, rotation, trans);
            AnalyticCollisionObject<T, dim> torus_CO(torus, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(torus_CO); // add ground
            // MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSet(torus, density, ppc);
            // CorotatedIsotropic<T, dim> model2(E, nu);
            // particles_handle.addFBasedMpmForce(model2);

            sim.output_dir.path = destination("output/LKCatBounceTorus", sim.step.max_dt, sim.dx, E[1], ppc[1], sim.particles.count);
        }

        /*
        ./multigrid -test 9910 -v_mu 2 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --bcproject --linesearch -mg_level 3 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 2 -smoother 5 -dbg 0 -o lbfgssnow
        ./multigrid -test 9910 -v_mu 2 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --bcproject --linesearch -mg_level 1 -mg_times 10000 -mg_scale 0 -mg_omega 1.0 -coarseSolver 2 -smoother 2 -dbg 0 -o lbfgshsnow
        ./multigrid -test 9910 -v_mu 2 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 -o pnsnow_mf --matfree
        ./multigrid -test 9910 -v_mu 2 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 -o pnsnow
        */
        if (test_number == 9910) {

            sim.verbose = true;
            sim.end_frame = 240;
            //sim.end_frame = 110;
            sim.dx = 0.005 * 0.7;
            sim.gravity = -2.5 * TV::Unit(1);

            T gmax = 1. / 24.;

            sim.step.max_dt = gmax;

            for (int i = 0; i < CaseSettings::v_mu; ++i)
                sim.step.max_dt /= 5.;

            sim.autorestart = false;
            sim.output_dir.path = "output/snowballfallgroundlinearclf0.6_dt" + std::to_string(sim.step.max_dt);

            //            sim.cfl = 0.05;
            sim.cfl = 0.6;
            sim.step.frame_dt = 1. / 120.;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
            sim.flip_pic_ratio = 0.95;

            sim.objective.minres.max_iterations = 10000; // max minres iteration
            sim.objective.minres.tolerance = 1e-7;
            sim.newton.max_iterations = 10000;
            sim.newton.tolerance = 1e-7;
            sim.quasistatic = false;
            sim.symplectic = false;
            sim.objective.matrix_free = false;

            const T mass_density = (T)2;
            const int number_of_particles = 300000 / (0.7 * 0.7 * 0.7);
            const Sphere<T, dim> ball(TV(0, 1.35, 0), .1);
            const Sphere<T, dim> padded_ball(ball.center, 0.11);
            const T ball_volume = ball.volume();
            const T volume_per_particle = ball_volume / number_of_particles;

            RandomNumber<T> random;
            StdVector<Sphere<T, dim>> spheres;
            for (int i = 1; i <= 500; i++) {
                TV offset = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius;
                if (offset.norm() < padded_ball.radius * 0.5 || offset.norm() > padded_ball.radius * 0.9) {
                    i--;
                    continue;
                }
                T radius = random.randReal(0, padded_ball.radius - offset.norm());
                if (radius < padded_ball.radius * 0.1) {
                    i--;
                    continue;
                }
                spheres.push_back(Sphere<T, dim>(offset + padded_ball.center, radius));
            }

            const T E = 100, nu = 0.25;

            const T h = 10;
            const T a = 1.025;
            const T ea = std::exp((a - 1) * h);

            struct ParticleTmp {
                TV X, V;
                T mass;
                T mu0, lambda0;
                T stretching_yield, compression_yield, hardening_factor;
            };
            StdVector<ParticleTmp> particles_buffer;

#define PERTURB(a) (random.randReal(1 - a, 1 + a))
#define UNIFORM(a, b) (random.randReal(a, b))
            for (int i = 1; i <= number_of_particles; i++) {
                ParticleTmp particle;
                particle.X = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius + padded_ball.center;

                bool inside = false;
                int count = 0;
                for (size_t j = 0; j < spheres.size(); j++) {
                    inside |= spheres[j].inside(particle.X);
                    count += spheres[j].inside(particle.X);
                }
                inside |= ball.inside(particle.X);
                if (!inside) {
                    i--;
                    continue;
                }

                particle.V = TV(0, -1.2, 0);
                particle.mass = mass_density * volume_per_particle * PERTURB(0.5);
                //particle.constitutive_model.Compute_Lame_Parameters(E*PERTURB(0.5),nu*PERTURB(0.1));

                {
                    T E2 = E * PERTURB(0.5);
                    T nu2 = nu * PERTURB(0.1);
                    particle.lambda0 = E2 * nu2 / (((T)1 + nu2) * ((T)1 - (T)2 * nu2));
                    particle.mu0 = E2 / ((T)2 * ((T)1 + nu2));
                }

                // different from disney
                particle.stretching_yield = .005 * PERTURB(0.01);
                particle.compression_yield = .025 * PERTURB(0.01);
                particle.hardening_factor = h * PERTURB(0.01);

                const int n = 10;
                T array[n] = { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 };

                int c = 0;
                for (int i = 0; i < n; i++)
                    if ((particle.X - ball.center).norm() > array[i] * ball.radius) {
                        c++;
                        particle.mass *= a;
                        particle.lambda0 *= ea;
                        particle.mu0 *= ea;
                    }

                T mult = 1;
                for (int i = 0; i < n - c; i++) mult *= (1 / a);

                if (UNIFORM(0, 1) > mult) continue;
                //mpm->particles.Append(particle);
                particles_buffer.push_back(particle);
            }

            for (int k = 1; k <= 500; k++) {
                const TV offset = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius * 5;
                const T radius = random.randReal(padded_ball.radius, padded_ball.radius * 10);
                Sphere<T, dim> sphere_big(offset + padded_ball.center, radius + sim.dx / 2 * 2);
                Sphere<T, dim> sphere_small(offset + padded_ball.center, radius - sim.dx / 2 * 2);
                const TV offset_new = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius * 5;
                const T radius_new = random.randReal(padded_ball.radius, padded_ball.radius * 10);
                Sphere<T, dim> sphere_new(offset_new + padded_ball.center, radius_new);
                if (offset_new.norm() + ball.radius < radius_new) continue;
                if ((offset - offset_new).norm() + radius < radius_new) continue;
                for (size_t i = 0; i < particles_buffer.size(); i++)
                    if (sphere_big.inside(particles_buffer[i].X) && !sphere_small.inside(particles_buffer[i].X) && sphere_new.inside(particles_buffer[i].X)) {
                        particles_buffer[i].lambda0 *= 0.5;
                        particles_buffer[i].mu0 *= 0.5;
                    }
            }

            auto velocity_point_to_right = [&](int index, Ref<T> mass, TV& X, TV& V) {
                TV omega(3, 1, 1);
                V(0) = 2 + omega(2) * X(1) - omega(1) * X(2);
                V(1) = omega(0) * X(2) - omega(2) * X(0);
                V(2) = omega(1) * X(0) - omega(0) * X(1);
            };
            for (size_t i = 0; i < particles_buffer.size(); ++i) {
                auto& particle = particles_buffer[i];
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleOneParticle(particle.X + TV(5, 5, 5), particle.V, particle.mass / volume_per_particle, volume_per_particle);
                //LinearCorotated<T, dim> model(1, 1);
                CorotatedIsotropic<T, dim> model(1, 1);
                model.project = true;
                model.mu = particle.mu0;
                model.lambda = particle.lambda0;
                SnowPlasticity<T> p(particle.hardening_factor, particle.compression_yield, particle.stretching_yield, 0, 20);
                //particles_handle.addLinearCorotatedMpmForce(model);
                particles_handle.addFBasedMpmForce(model);
                particles_handle.addPlasticity(model, p, "F");
            }

            // ground
            TV ground_origin(0 + 5, .1 + 5, 0 + 5);
            TV ground_normal(0, 0.6, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            //ground_object.setFriction(.4);
            init_helper.addAnalyticCollisionObject(ground_object);
        }

        if (test_number == 9911) {

            sim.verbose = true;
            sim.end_frame = 240;
            sim.dx = 0.005 * 0.7;
            sim.gravity = -2.5 * TV::Unit(1);

            T gmax = 1. / 24.;

            sim.step.max_dt = gmax;

            for (int i = 1; i < CaseSettings::v_mu; ++i)
                sim.step.max_dt /= 5.;

            sim.autorestart = false;
            sim.output_dir.path = "output/snowballfallgroundlinearclf0.6_dt" + std::to_string(sim.step.max_dt);

            //            sim.cfl = 0.05;
            sim.cfl = 1.0;
            sim.step.frame_dt = 1. / 120.;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
            sim.flip_pic_ratio = 0.95;

            sim.objective.minres.max_iterations = 10000; // max minres iteration
            sim.objective.minres.tolerance = 1e-7;
            sim.objective.cg.max_iterations = 10000; // max minres iteration
            sim.objective.cg.tolerance = 1e-7;
            sim.newton.max_iterations = 10000;
            sim.newton.tolerance = 1e-7;
            sim.quasistatic = false;
            sim.symplectic = false;
            sim.objective.matrix_free = HOTSettings::matrixFree;

            const T mass_density = (T)2;
            const int number_of_particles = 300000 / (0.7 * 0.7 * 0.7);
            const Sphere<T, dim> ball(TV(0, 1.35, 0), .1);
            const Sphere<T, dim> padded_ball(ball.center, 0.11);
            const T ball_volume = ball.volume();
            const T volume_per_particle = ball_volume / number_of_particles;

            RandomNumber<T> random;
            StdVector<Sphere<T, dim>> spheres;
            for (int i = 1; i <= 500; i++) {
                TV offset = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius;
                if (offset.norm() < padded_ball.radius * 0.5 || offset.norm() > padded_ball.radius * 0.9) {
                    i--;
                    continue;
                }
                T radius = random.randReal(0, padded_ball.radius - offset.norm());
                if (radius < padded_ball.radius * 0.1) {
                    i--;
                    continue;
                }
                spheres.push_back(Sphere<T, dim>(offset + padded_ball.center, radius));
            }

            const T E = 100, nu = 0.25;

            const T h = 10;
            const T a = 1.025;
            const T ea = std::exp((a - 1) * h);

            struct ParticleTmp {
                TV X, V;
                T mass;
                T mu0, lambda0;
                T stretching_yield, compression_yield, hardening_factor;
            };
            StdVector<ParticleTmp> particles_buffer;

#define PERTURB(a) (random.randReal(1 - a, 1 + a))
#define UNIFORM(a, b) (random.randReal(a, b))
            for (int i = 1; i <= number_of_particles; i++) {
                ParticleTmp particle;
                particle.X = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius + padded_ball.center;

                bool inside = false;
                int count = 0;
                for (size_t j = 0; j < spheres.size(); j++) {
                    inside |= spheres[j].inside(particle.X);
                    count += spheres[j].inside(particle.X);
                }
                inside |= ball.inside(particle.X);
                if (!inside) {
                    i--;
                    continue;
                }

                particle.V = TV(0, -1.2, 0);
                particle.mass = mass_density * volume_per_particle * PERTURB(0.5);
                //particle.constitutive_model.Compute_Lame_Parameters(E*PERTURB(0.5),nu*PERTURB(0.1));

                {
                    T E2 = E * PERTURB(0.5);
                    T nu2 = nu * PERTURB(0.1);
                    particle.lambda0 = E2 * nu2 / (((T)1 + nu2) * ((T)1 - (T)2 * nu2));
                    particle.mu0 = E2 / ((T)2 * ((T)1 + nu2));
                }

                // different from disney
                particle.stretching_yield = .005 * PERTURB(0.01);
                particle.compression_yield = .025 * PERTURB(0.01);
                particle.hardening_factor = h * PERTURB(0.01);

                const int n = 10;
                T array[n] = { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 };

                int c = 0;
                for (int i = 0; i < n; i++)
                    if ((particle.X - ball.center).norm() > array[i] * ball.radius) {
                        c++;
                        particle.mass *= a;
                        particle.lambda0 *= ea;
                        particle.mu0 *= ea;
                    }

                T mult = 1;
                for (int i = 0; i < n - c; i++) mult *= (1 / a);

                if (UNIFORM(0, 1) > mult) continue;
                //mpm->particles.Append(particle);
                particles_buffer.push_back(particle);
            }

            for (int k = 1; k <= 500; k++) {
                const TV offset = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius * 5;
                const T radius = random.randReal(padded_ball.radius, padded_ball.radius * 10);
                Sphere<T, dim> sphere_big(offset + padded_ball.center, radius + sim.dx / 2 * 2);
                Sphere<T, dim> sphere_small(offset + padded_ball.center, radius - sim.dx / 2 * 2);
                const TV offset_new = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius * 5;
                const T radius_new = random.randReal(padded_ball.radius, padded_ball.radius * 10);
                Sphere<T, dim> sphere_new(offset_new + padded_ball.center, radius_new);
                if (offset_new.norm() + ball.radius < radius_new) continue;
                if ((offset - offset_new).norm() + radius < radius_new) continue;
                for (size_t i = 0; i < particles_buffer.size(); i++)
                    if (sphere_big.inside(particles_buffer[i].X) && !sphere_small.inside(particles_buffer[i].X) && sphere_new.inside(particles_buffer[i].X)) {
                        particles_buffer[i].lambda0 *= 0.5;
                        particles_buffer[i].mu0 *= 0.5;
                    }
            }

            auto velocity_point_to_right = [&](int index, Ref<T> mass, TV& X, TV& V) {
                TV omega(3, 1, 1);
                V(0) = 2 + omega(2) * X(1) - omega(1) * X(2);
                V(1) = omega(0) * X(2) - omega(2) * X(0);
                V(2) = omega(1) * X(0) - omega(0) * X(1);
            };
            for (size_t i = 0; i < particles_buffer.size(); ++i) {
                auto& particle = particles_buffer[i];
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleOneParticle(particle.X + TV(5, 5, 5), particle.V + TV(0, 0, 0), particle.mass / volume_per_particle, volume_per_particle);
                LinearCorotated<T, dim> model(1, 1);
                model.mu = particle.mu0;
                model.lambda = particle.lambda0;
                SnowPlasticity<T> p(particle.hardening_factor, particle.compression_yield, particle.stretching_yield);
                particles_handle.addLinearCorotatedMpmForce(model);
                particles_handle.addPlasticity(model, p, "F");
            }

            // ground
            TV ground_origin(0 + 5, 0.1 + 5, 0 + 5);
            TV ground_normal(0, 0.6, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            //ground_object.setFriction(.4);
            init_helper.addAnalyticCollisionObject(ground_object);

            sim.output_dir.path = destination("output/FCR_snowballdrop", sim.step.max_dt, sim.dx, E, 0, sim.particles.count);
        }

        if (test_number == 777001) { // JOB bar twist
            /*
             #!/bin/bash
            declare -a StringArray=("1e4" "1e5" "1e6" "1e7" "1e8" "1e9" "1e10" "1e11")
            for val in ${StringArray[@]}; do
                ./multigrid -test 777001 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -cmd0 $val -o job-ours-E$val
                ./multigrid -test 777001 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -cmd0 $val -o job-pn-E$val
                ./multigrid -test 777001 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --matfree   -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -cmd0 $val -o job-pnfree-E$val
            done
            */
            sim.end_frame = 48;
            sim.dx = 0.0075;
            sim.gravity = -0 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)48;
            sim.step.max_dt = sim.step.frame_dt;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 1;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            int ppc = 12;
            T half = 0.06;
            RandomNumber<T> random;
            {
                AxisAlignedAnalyticBox<T, dim> box(TV(2 - half, 2.8, 2 - half), TV(2 + half, 3.1, 2 + half));
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(box, 2e3, ppc);
                CorotatedIsotropic<T, dim> model(1e5, .3);
                particles_handle.addFBasedMpmForce(model);
            }
            {
                AxisAlignedAnalyticBox<T, dim> box(TV(2 - half, 3.1, 2 - half), TV(2 + half, 3.4, 2 + half));
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(box, 2e3, ppc);
                CorotatedIsotropic<T, dim> model(CmdArgument::cmd0, .3);
                // VonMisesFixedCorotated<T, dim> p(240e6);
                // particles_handle.addPlasticity(model, p, "F");
                particles_handle.addFBasedMpmForce(model);
            }
            {
                AxisAlignedAnalyticBox<T, dim> box(TV(2 - half, 3.4, 2 - half), TV(2 + half, 3.7, 2 + half));
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(box, 2e3, ppc);
                CorotatedIsotropic<T, dim> model(1e5, .3);
                particles_handle.addFBasedMpmForce(model);
            }

            for (int c = 0; c < 2; c++) {
                T y = c ? 3.7 : 2.8;
                T s = c ? 1 : 0;
                T rise_speed = c ? 0 : 0;
                auto ceilTransform = [y, s, rise_speed](T time, AnalyticCollisionObject<T, dim>& object) {
                    T theta = s * (T)360.0 / 180 * M_PI;
                    T t = time;
                    if (t < 1) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, theta, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, rise_speed, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = 1;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                T theta = 0;
                Vector<T, 4> rotation(std::cos(theta / 2), std::sin(theta / 2), 0, 0);
                TV translation(0, 0, 0);
                CappedCylinder<T, dim> cylinder(/*radius*/ .2, /*height*/ .1, rotation, translation);
                AnalyticCollisionObject<T, dim> ceilObject(ceilTransform, cylinder, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(ceilObject);
            }
        }

        if (test_number == 777002) {
            //./multigrid -test 777002 --3d --usecn -cneps 0.000001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o test_777002ours
            //./multigrid -test 777002 --3d --usecn -cneps 0.000001 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 -o test_777002pn
            //sim.output_dir.path = "output/debug";
            sim.end_frame = 148;

            sim.dx = 0.0075; // 0.005 for final example

            sim.gravity = -9 * TV::Unit(1);

            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = 1e7; ///< use absolute tolerance

            //T rho = 2;
            //T nu = 0.4, youngs = 500;
            T nu = 0.4;
            T rho_soft_rubber = 2e3;
            T youngs_soft_rubber = CmdArgument::cmd1 > 0 ? CmdArgument::cmd1 : 5e5;
            T rho_wood = 2e3;
            T youngs_wood = CmdArgument::cmd0 > 0 ? CmdArgument::cmd0 : 3e9; //1e7;
            int ppc = 8;

            sim.output_dir.path = destination("output/FCRtimingFractureTwist", sim.step.max_dt, sim.dx, youngs_soft_rubber, ppc, sim.particles.count);

            T half = 0.04;
            RandomNumber<T> random;
            for (int i = -1; i <= 1; i += 2) {
                for (int j = -1; j <= 1; j += 2) {
                    T dx = i * 0.085, dz = j * 0.085;
                    AxisAlignedAnalyticBox<T, dim> box(TV(2 - half, 2.8, 2 - half) + TV(dx, 0, dz), TV(2 + half, 3.7, 2 + half) + TV(dx, 0, dz));
                    MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(box, rho_soft_rubber, ppc);
                    CorotatedIsotropic<T, dim> model(youngs_soft_rubber, nu);
                    model.project = HOTSettings::project;
                    particles_handle.addFBasedMpmForce(model);
                }
            }
            ZIRAN_INFO("pillar count: ", sim.particles.count);
            int count = 0;
            for (int i = -1; i <= 1; i += 2) {
                for (int j = -1; j <= 1; j += 2) {
                    T dx = i * 0.085, dz = j * 0.085;
                    T ss[4] = { 0.25, 0.65, 0.55, 0.45 };
                    TV center(2 + dx, 2.8 + ss[count++], 2 + dz);
                    T radius = half * (2 + random.randReal(-0.2, 0.3));
                    Sphere<T, dim> sphere(center, radius);
                    MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSet(sphere, rho_wood, ppc);
                    CorotatedIsotropic<T, dim> model2(youngs_wood, nu);
                    model2.project = HOTSettings::project;
                    particles_handle2.addFBasedMpmForce(model2);
                }
            }

            // auto ptrans = [](int index, Ref<T> mass, TV& X, TV& V) { X += TV(0, 0, 0); };
            // particles_handle.transform(ptrans);

            for (int c = 0; c < 2; c++) {
                T y = c ? 3.7 : 2.8;
                T s = c ? 1 : -1;
                T rise_speed = c ? 0.1 : 0; // 0.1 for final example
                auto ceilTransform = [y, s, rise_speed](T time, AnalyticCollisionObject<T, dim>& object) {
                    T theta = s * (T)100.0 / 180 * M_PI; // 100 for final example
                    T t = time;
                    if (t < 3) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, theta, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, rise_speed, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = 3;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                T theta = 0;
                Vector<T, 4> rotation(std::cos(theta / 2), std::sin(theta / 2), 0, 0);
                TV translation(0, 0, 0);
                CappedCylinder<T, dim> cylinder(/*radius*/ .2, /*height*/ .1, rotation, translation);
                AnalyticCollisionObject<T, dim> ceilObject(ceilTransform, cylinder, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(ceilObject);
            }

            //Set up a ground plane
            TV ground_origin(0, 2.75, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            ground_object.setFriction(1000.0);
            init_helper.addAnalyticCollisionObject(ground_object); // add ground
        }

        if (test_number == 777003) {
            sim.output_dir.path = "output/snowballs_colliding";
            sim.end_frame = 120;
            T dx_scale = 1;
            sim.dx = 0.005 * dx_scale;
            sim.gravity = -2.5 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = 0.005;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
            sim.apic_rpic_ratio = 0.95;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.5;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            for (int ballcount = 0; ballcount < 2; ballcount++) {
                const T mass_density = (T)2;
                const int number_of_particles = 300000 / std::pow(dx_scale, 3);
                const Sphere<T, dim> ball(ballcount ? TV(1.85, .52, .5) : TV(-.85, .523, .502), .1);
                const Sphere<T, dim> padded_ball(ball.center, 0.11);
                const T ball_volume = ball.volume();
                const T volume_per_particle = ball_volume / number_of_particles;

                RandomNumber<T> random;
                StdVector<Sphere<T, dim>> spheres;
                for (int i = 1; i <= 500; i++) {
                    TV offset = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius;
                    if (offset.norm() < padded_ball.radius * 0.5 || offset.norm() > padded_ball.radius * 0.9) {
                        i--;
                        continue;
                    }
                    T radius = random.randReal(0, padded_ball.radius - offset.norm());
                    if (radius < padded_ball.radius * 0.1) {
                        i--;
                        continue;
                    }
                    spheres.push_back(Sphere<T, dim>(offset + padded_ball.center, radius));
                }

                const T E = 100, nu = 0.25;

                const T h = 10;
                const T a = 1.025;
                const T ea = std::exp((a - 1) * h);

                struct ParticleTmp {
                    TV X, V;
                    T mass;
                    T mu0, lambda0;
                    T stretching_yield, compression_yield, hardening_factor;
                };
                StdVector<ParticleTmp> particles_buffer;

#define PERTURB(a) (random.randReal(1 - a, 1 + a))
#define UNIFORM(a, b) (random.randReal(a, b))

                const TV angular_1 = TV(UNIFORM(-5, 5), UNIFORM(-5, 5), UNIFORM(-5, 5));
                const TV angular_0 = TV(UNIFORM(-5, 5), UNIFORM(-5, 5), UNIFORM(-5, 5));

                const TV velocity_1 = TV(-3 * PERTURB(0.05), 0.75 * PERTURB(0.2), UNIFORM(-0.2, 0.2));
                const TV velocity_0 = TV(3 * PERTURB(0.05), 0.75 * PERTURB(0.2), UNIFORM(-0.2, 0.2));

                for (int i = 1; i <= number_of_particles; i++) {
                    ParticleTmp particle;
                    particle.X = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius + padded_ball.center;

                    bool inside = false;
                    int count = 0;
                    for (size_t j = 0; j < spheres.size(); j++) {
                        inside |= spheres[j].inside(particle.X);
                        count += spheres[j].inside(particle.X);
                    }
                    inside |= ball.inside(particle.X);
                    if (!inside) {
                        i--;
                        continue;
                    }

                    particle.V = ballcount ? velocity_1 : velocity_0;
                    particle.V += (ballcount ? angular_1 : angular_0).cross(particle.X - ball.center);
                    particle.mass = mass_density * volume_per_particle * PERTURB(0.5);
                    //particle.constitutive_model.Compute_Lame_Parameters(E*PERTURB(0.5),nu*PERTURB(0.1));

                    {
                        T E2 = E * PERTURB(0.5);
                        T nu2 = nu * PERTURB(0.1);
                        particle.lambda0 = E2 * nu2 / (((T)1 + nu2) * ((T)1 - (T)2 * nu2));
                        particle.mu0 = E2 / ((T)2 * ((T)1 + nu2));
                    }

                    // different from disney
                    particle.stretching_yield = .005 * PERTURB(0.01);
                    particle.compression_yield = .025 * PERTURB(0.01);
                    particle.hardening_factor = h * PERTURB(0.01);

                    const int n = 10;
                    T array[n] = { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 };

                    int c = 0;
                    for (int i = 0; i < n; i++)
                        if ((particle.X - ball.center).norm() > array[i] * ball.radius) {
                            c++;
                            particle.mass *= a;
                            particle.lambda0 *= ea;
                            particle.mu0 *= ea;
                        }

                    T mult = 1;
                    for (int i = 0; i < n - c; i++) mult *= (1 / a);

                    if (UNIFORM(0, 1) > mult) continue;
                    //mpm->particles.Append(particle);
                    particles_buffer.push_back(particle);
                }

                for (int k = 1; k <= 500; k++) {
                    const TV offset = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius * 5;
                    const T radius = random.randReal(padded_ball.radius, padded_ball.radius * 10);
                    Sphere<T, dim> sphere_big(offset + padded_ball.center, radius + sim.dx / 2 * 2);
                    Sphere<T, dim> sphere_small(offset + padded_ball.center, radius - sim.dx / 2 * 2);
                    const TV offset_new = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius * 5;
                    const T radius_new = random.randReal(padded_ball.radius, padded_ball.radius * 10);
                    Sphere<T, dim> sphere_new(offset_new + padded_ball.center, radius_new);
                    if (offset_new.norm() + ball.radius < radius_new) continue;
                    if ((offset - offset_new).norm() + radius < radius_new) continue;
                    for (size_t i = 0; i < particles_buffer.size(); i++)
                        if (sphere_big.inside(particles_buffer[i].X) && !sphere_small.inside(particles_buffer[i].X) && sphere_new.inside(particles_buffer[i].X)) {
                            particles_buffer[i].lambda0 *= 0.5;
                            particles_buffer[i].mu0 *= 0.5;
                        }
                }

                for (size_t i = 0; i < particles_buffer.size(); ++i) {
                    auto& particle = particles_buffer[i];
                    MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleOneParticle(particle.X + TV(5, 5, 5), particle.V + TV(0, 0, 0), particle.mass / volume_per_particle, volume_per_particle);
                    CorotatedIsotropic<T, dim> model(1, 1);
                    model.mu = particle.mu0;
                    model.lambda = particle.lambda0;
                    SnowPlasticity<T> p(particle.hardening_factor, particle.compression_yield, particle.stretching_yield);
                    particles_handle.addFBasedMpmForce(model);
                    particles_handle.addPlasticity(model, p, "F");
                }
            }

            TV ground_origin(0 + 5, 0.1 + 5, 0 + 5);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);
        }

        if (test_number == 777004) { // snowball hit ground
            /*
            ./multigrid -test 777004 --3d --usecn -cneps 0.000001 -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o pn
            ./multigrid -test 777004 --3d --usecn -cneps 0.000001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours
            ./multigrid -test 777004 --3d --usecn -cneps 0.000001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 --adaptiveH -o ours-adaptiveH
        */
            sim.verbose = true;
            sim.end_frame = 240;
            sim.dx = 0.005;
            sim.gravity = -2.5 * TV::Unit(1);
            sim.step.max_dt = 0.005;
            sim.autorestart = false;
            sim.output_dir.path = "output/snowball";
            sim.cfl = 0.6;
            sim.step.frame_dt = 1. / 24.;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
            sim.flip_pic_ratio = 0.98;

            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-7;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-7;
            sim.newton.max_iterations = 1;
            sim.newton.tolerance = 1e-7;
            sim.quasistatic = false;
            sim.symplectic = false;
            sim.objective.matrix_free = false;

            const T mass_density = (T)2;
            const int number_of_particles = 300000;
            const Sphere<T, dim> ball(TV(0, 1.35, 0), .1);
            const Sphere<T, dim> padded_ball(ball.center, 0.11);
            const T ball_volume = ball.volume();
            const T volume_per_particle = ball_volume / number_of_particles;

            RandomNumber<T> random;
            StdVector<Sphere<T, dim>> spheres;
            for (int i = 1; i <= 500; i++) {
                TV offset = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius;
                if (offset.norm() < padded_ball.radius * 0.5 || offset.norm() > padded_ball.radius * 0.9) {
                    i--;
                    continue;
                }
                T radius = random.randReal(0, padded_ball.radius - offset.norm());
                if (radius < padded_ball.radius * 0.1) {
                    i--;
                    continue;
                }
                spheres.push_back(Sphere<T, dim>(offset + padded_ball.center, radius));
            }

            const T E = 100, nu = 0.25;

            const T h = 10;
            const T a = 1.025;
            const T ea = std::exp((a - 1) * h);

            struct ParticleTmp {
                TV X, V;
                T mass;
                T mu0, lambda0;
                T stretching_yield, compression_yield, hardening_factor;
            };
            StdVector<ParticleTmp> particles_buffer;

#define PERTURB(a) (random.randReal(1 - a, 1 + a))
#define UNIFORM(a, b) (random.randReal(a, b))
            for (int i = 1; i <= number_of_particles; i++) {
                ParticleTmp particle;
                particle.X = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius + padded_ball.center;

                bool inside = false;
                int count = 0;
                for (size_t j = 0; j < spheres.size(); j++) {
                    inside |= spheres[j].inside(particle.X);
                    count += spheres[j].inside(particle.X);
                }
                inside |= ball.inside(particle.X);
                if (!inside) {
                    i--;
                    continue;
                }

                particle.V = TV(0, -1.2, 0);
                particle.mass = mass_density * volume_per_particle * PERTURB(0.5);
                //particle.constitutive_model.Compute_Lame_Parameters(E*PERTURB(0.5),nu*PERTURB(0.1));

                {
                    T E2 = E * PERTURB(0.5);
                    T nu2 = nu * PERTURB(0.1);
                    particle.lambda0 = E2 * nu2 / (((T)1 + nu2) * ((T)1 - (T)2 * nu2));
                    particle.mu0 = E2 / ((T)2 * ((T)1 + nu2));
                }

                // different from disney
                particle.stretching_yield = .005 * PERTURB(0.01);
                particle.compression_yield = .025 * PERTURB(0.01);
                particle.hardening_factor = h * PERTURB(0.01);

                const int n = 10;
                T array[n] = { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 };

                int c = 0;
                for (int i = 0; i < n; i++)
                    if ((particle.X - ball.center).norm() > array[i] * ball.radius) {
                        c++;
                        particle.mass *= a;
                        particle.lambda0 *= ea;
                        particle.mu0 *= ea;
                    }

                T mult = 1;
                for (int i = 0; i < n - c; i++) mult *= (1 / a);

                if (UNIFORM(0, 1) > mult) continue;
                //mpm->particles.Append(particle);
                particles_buffer.push_back(particle);
            }

            for (int k = 1; k <= 500; k++) {
                const TV offset = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius * 5;
                const T radius = random.randReal(padded_ball.radius, padded_ball.radius * 10);
                Sphere<T, dim> sphere_big(offset + padded_ball.center, radius + sim.dx / 2 * 2);
                Sphere<T, dim> sphere_small(offset + padded_ball.center, radius - sim.dx / 2 * 2);
                const TV offset_new = random.randInBall(TV(0, 0, 0), 1) * padded_ball.radius * 5;
                const T radius_new = random.randReal(padded_ball.radius, padded_ball.radius * 10);
                Sphere<T, dim> sphere_new(offset_new + padded_ball.center, radius_new);
                if (offset_new.norm() + ball.radius < radius_new) continue;
                if ((offset - offset_new).norm() + radius < radius_new) continue;
                for (size_t i = 0; i < particles_buffer.size(); i++)
                    if (sphere_big.inside(particles_buffer[i].X) && !sphere_small.inside(particles_buffer[i].X) && sphere_new.inside(particles_buffer[i].X)) {
                        particles_buffer[i].lambda0 *= 0.5;
                        particles_buffer[i].mu0 *= 0.5;
                    }
            }

            auto velocity_point_to_right = [&](int index, Ref<T> mass, TV& X, TV& V) {
                TV omega(3, 1, 1);
                V(0) = 2 + omega(2) * X(1) - omega(1) * X(2);
                V(1) = omega(0) * X(2) - omega(2) * X(0);
                V(2) = omega(1) * X(0) - omega(0) * X(1);
            };
            for (size_t i = 0; i < particles_buffer.size(); ++i) {
                auto& particle = particles_buffer[i];
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleOneParticle(particle.X + TV(5, 5, 5), particle.V + TV(0, 0, 0), particle.mass / volume_per_particle, volume_per_particle);
                CorotatedIsotropic<T, dim> model(1, 1);
                model.project = HOTSettings::project;
                model.mu = particle.mu0;
                model.lambda = particle.lambda0;
                SnowPlasticity<T> p(particle.hardening_factor, particle.compression_yield, particle.stretching_yield, 0.001, 20);
                particles_handle.addFBasedMpmForce(model);
                particles_handle.addPlasticity(model, p, "F");
            }

            // ground
            TV ground_origin(0 + 5, 0.1 + 5, 0 + 5);
            TV ground_normal(0, 0.6, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            //ground_object.setFriction(.4);
            init_helper.addAnalyticCollisionObject(ground_object);
        }

        if (test_number == 777005) {
            sim.end_frame = 48;

            sim.dx = 0.015;

            sim.gravity = -9.8 * TV::Unit(1);

            sim.step.frame_dt = (T)1 / (T)24;
            sim.cfl = .6;

            sim.step.max_dt = (T)1 / (T)24.0;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 1;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = 1e7; ///< use absolute tolerance

            T nu = 0.4;
            int ppc = 8;

            // sim.output_dir.path = destination("output/FCRtimingFractureTwist", sim.step.max_dt, sim.dx, youngs, ppc, sim.particles.count);
            sim.output_dir.path = "output/shoot";

            AxisAlignedAnalyticBox<T, dim> box(TV(-0.025, -0.5, -0.5), TV(0.025, 0.5, 0.5));
            MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(box, 1522, ppc);
            CorotatedIsotropic<T, dim> model(0.001 * 1e9, nu);
            model.project = HOTSettings::project;
            particles_handle.addFBasedMpmForce(model);
            particles_handle.transform([](int index, Ref<T> mass, TV& X, TV& V) {
                X += TV(1, 1, 1);
            });

            {
                TV sphere_center1(-0.4, 0, 0);
                Sphere<T, dim> sphere1(sphere_center1, 0.05);
                MpmParticleHandleBase<T, dim> particles_handle_top = init_helper.sampleInAnalyticLevelSet(sphere1, 8050, ppc);
                CorotatedIsotropic<T, dim> model2(20 * 1e9, nu);
                model2.project = HOTSettings::project;
                particles_handle_top.addFBasedMpmForce(model2);
                particles_handle_top.transform([](int index, Ref<T> mass, TV& X, TV& V) {
                    X += TV(1, 1, 1);
                    V = TV(10, 0, 0);
                });
            }

            AxisAlignedAnalyticBox<T, dim> b1(TV(-.05, -.05, -.6) + TV(1, 1.5, 1), TV(.05, .05, .6) + TV(1, 1.5, 1));
            AnalyticCollisionObject<T, dim> box_1(b1, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(box_1);

            AxisAlignedAnalyticBox<T, dim> b2(TV(-.05, -.05, -.6) + TV(1, .5, 1), TV(.05, .05, .6) + TV(1, .5, 1));
            AnalyticCollisionObject<T, dim> box_2(b2, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(box_2);

            TV ground_origin(0, .25, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);
        }

        if (test_number == 777006) { // snow wedge
            sim.end_frame = 120;

            sim.dx = 0.014;

            sim.gravity = -9.8 * TV::Unit(1);

            sim.step.frame_dt = (T)1 / (T)24;
            sim.cfl = .6;

            sim.step.max_dt = (T)1 / (T)24.0;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
            sim.flip_pic_ratio = 0.95;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = 1e7; ///< use absolute tolerance

            int ppc = 8;
            T rho = 400;
            T E = 1.4e5;
            T nu = 0.2;
            T theta_c = 2.5e-2;
            T theta_s = 7.5e-3;
            T hardening = 10;

            sim.output_dir.path = "output/snow_wedge";

            TV min_corner(-0.2 + 1, .7 + 1, -0.1 + 1);
            TV max_corner(0.2 + 1, .9 + 1, 0.1 + 1);
            AxisAlignedAnalyticBox<T, dim> box(min_corner, max_corner);
            MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSetFullyRandom(box, rho, 8);
            sim.recomputeLocationBasedElementMeasure = true; // refresh volume
            CorotatedIsotropic<T, dim> model(E, nu);
            model.project = HOTSettings::project;
            SnowPlasticity<T> p(hardening, theta_c, theta_s, /*minJp*/ 0.1, /*maxJp*/ 50);
            particles_handle.addFBasedMpmForce(model);
            particles_handle.addPlasticity(model, p, "F");

            {
                TV sphere_center1(1, 2, 1);
                Sphere<T, dim> sphere1(sphere_center1, 0.05);
                MpmParticleHandleBase<T, dim> particles_handle_top = init_helper.sampleInAnalyticLevelSet(sphere1, rho, ppc);
                CorotatedIsotropic<T, dim> model2(1e8, nu);
                model2.project = HOTSettings::project;
                particles_handle_top.addFBasedMpmForce(model2);
            }

            // Ground is at 0.1;
            TV ground_origin(0 + 1, 0.1 + 1, 0 + 1);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);

            // Wedge;
            TV half_edge(0.1, 0.1, 0.18);
            T theta = (T)135 / 180 * M_PI;
            Vector<T, 4> rotation(std::cos(theta / 2), 0, 0, std::sin(theta / 2));
            TV translation(0.0 + 1, 0.4 + 1, 0.0 + 1);
            AnalyticBox<T, dim> wedge(half_edge, rotation, translation);
            AnalyticCollisionObject<T, dim> wedge_object(wedge, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(wedge_object);
        }

        if (test_number == 777007) {
            //./multigrid -test 777007 --3d --usecn -cneps 0.000001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 --l2norm -o test777007_ours
            //./multigrid -test 777007 --3d --usecn -cneps 0.000001 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 --l2norm -o test777007_pn
            sim.end_frame = 60;

            sim.dx = 0.01;

            sim.gravity = -1 * TV::Unit(1);

            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = std::sqrt(HOTSettings::characterNorm); // extended gast15

            T density[2] = { 2500, 1000 };
            T E[2] = { 3e9, 2.5e4 };
            T nu = 0.4;
            T rho_soft_rubber = 2e3;
            T youngs_soft_rubber = CmdArgument::cmd1 > 0 ? CmdArgument::cmd1 : 5e5;
            //T rho_wood = 2e3;
            //T youngs_wood = 1e8;
            //T rho_soft_rubber = density[1];
            //T youngs_soft_rubber = E[1];
            T rho_wood = density[0];
            T youngs_wood = CmdArgument::cmd0 > 0 ? CmdArgument::cmd0 : E[0];
            int ppc = 8;

            //sim.output_dir.path = destination("output/FCRtimingFractureTwist", sim.step.max_dt, sim.dx, youngs, ppc, sim.particles.count);

            T half = 0.04;
            RandomNumber<T> random;
            for (int i = -1; i <= 1; i += 2) {
                for (int j = -1; j <= 1; j += 2) {
                    T dx = i * 0.085, dz = j * 0.085;
                    AxisAlignedAnalyticBox<T, dim> box(TV(2 - half, 2.8, 2 - half) + TV(dx, 0, dz), TV(2 + half, 3.7, 2 + half) + TV(dx, 0, dz));
                    MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(box, rho_soft_rubber, ppc);
                    CorotatedIsotropic<T, dim> model(youngs_soft_rubber, nu);
                    model.project = HOTSettings::project;
                    particles_handle.addFBasedMpmForce(model);
                }
            }
            ZIRAN_INFO("pillar count: ", sim.particles.count);
            int count = 0;
            for (int i = -1; i <= 1; i += 2) {
                for (int j = -1; j <= 1; j += 2) {
                    T dx = i * 0.085, dz = j * 0.085;
                    T ss[4] = { 0.25, 0.65, 0.55, 0.45 };
                    TV center(2 + dx, 2.8 + ss[count++], 2 + dz);
                    T radius = half * (2 + random.randReal(-0.2, 0.3));
                    Sphere<T, dim> sphere(center, radius);
                    MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSet(sphere, rho_wood, ppc);
                    CorotatedIsotropic<T, dim> model2(youngs_wood, nu);
                    model2.project = HOTSettings::project;
                    particles_handle2.addFBasedMpmForce(model2);
                }
            }

            for (int c = 0; c < 2; c++) {
                T y = c ? 3.7 : 2.8;
                T s = c ? 1 : -1;
                T rise_speed = c ? 0 : 0;
                auto ceilTransform = [y, s, rise_speed](T time, AnalyticCollisionObject<T, dim>& object) {
                    T theta = s * (T)100.0 / 180 * M_PI; // 100 for final example
                    T t = time;
                    if (t < 1.6667) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, theta, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, rise_speed, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = 1.6667;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                T theta = 0;
                Vector<T, 4> rotation(std::cos(theta / 2), std::sin(theta / 2), 0, 0);
                TV translation(0, 0, 0);
                CappedCylinder<T, dim> cylinder(/*radius*/ .2, /*height*/ .1, rotation, translation);
                AnalyticCollisionObject<T, dim> ceilObject(ceilTransform, cylinder, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(ceilObject);
            }

            //Set up a ground plane
            TV ground_origin(0, 2.75, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            ground_object.setFriction(1000.0);
            init_helper.addAnalyticCollisionObject(ground_object); // add ground
        }

        if (test_number == 777008) {
            /*
            ./multigrid -test 777008 --3d --usecn -cneps 0.000001 -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o pn
            ./multigrid -test 777008 --3d --usecn -cneps 0.000001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours
            ./multigrid -test 777008 --3d --usecn -cneps 0.000001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 --adaptiveH -o ours-ming
            */
            sim.end_frame = 85;
            sim.dx = 0.005;
            sim.gravity = -9.81 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt / 2;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6 / 2;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            int ppc = 8;

            if (1) {
                // ABS plastic
                T nu = 0.35;
                T rho = 1.53e3;
                T E = 1.4e9;
                T yield = 4.82e7;

                // Aluminum
                // T nu = 0.33;
                // T rho = 2.7e3;
                // T E = 69e9;
                // T yield = 240e6;

                // RandomNumber<T> random;
                // StdVector<TV> meshed_points;
                // readPositionObj("armadillos1.obj", meshed_points);
                // MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/armadillos1.vdb", rho, ppc);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFile("LevelSets/armadillos1.vdb", rho, ppc);

                CorotatedIsotropic<T, dim> model(E, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
                VonMisesFixedCorotated<T, dim> p(yield);
                particles_handle.addPlasticity(model, p, "F");
            }
            {
                // ABS plastic
                T nu = 0.35;
                T rho = 1.53e3;
                T E = 1.4e9;
                T yield = 4.82e7;

                // RandomNumber<T> random;
                // StdVector<TV> meshed_points;
                // readPositionObj("armadillos2.obj", meshed_points);
                // MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/armadillos2.vdb", rho, ppc);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFile("LevelSets/armadillos2.vdb", rho, ppc);
                CorotatedIsotropic<T, dim> model(E, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
                // VonMisesFixedCorotated<T, dim> p(yield);
                // particles_handle.addPlasticity(model, p, "F");
            }

            auto ceilTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
                T t = time;
                T shrink_speed = 0.2;
                T shrink_time = 3.5;
                T hold_time = 3.7;
                ZIRAN_ASSERT(1 - shrink_speed * shrink_time > 0);
                if (t < shrink_time) {
                    TV translation_velocity(0, 0, 0);
                    TV translation(0.53, 0.45, 0.54);
                    object.setTranslation(translation, translation_velocity);
                    object.setScaling(1 - shrink_speed * t, -shrink_speed);
                }
                else if (t < hold_time) {
                    t = shrink_time;
                    TV translation_velocity(0, 0, 0);
                    TV translation(0.53, 0.45, 0.54);
                    object.setTranslation(translation, translation_velocity);
                    object.setScaling(1 - shrink_speed * t, 0);
                }
                else {
                    t -= hold_time;
                    TV translation_velocity(0, 0, 0);
                    TV translation(0.53, 0.45, 0.54);
                    object.setTranslation(translation, translation_velocity);
                    object.setScaling(1 - shrink_speed * shrink_time + shrink_speed * t, shrink_speed);
                }
            };

            TV center(0, 0, 0);
            Sphere<T, dim> sphere1(center, 0.4);
            Sphere<T, dim> sphere2(center, 0.15);
            DifferenceLevelSet<T, dim> materialRegionLevelSet;
            materialRegionLevelSet.add(sphere1, sphere2);
            AnalyticCollisionObject<T, dim> ceilObject(ceilTransform, materialRegionLevelSet, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ceilObject);
        }

        if (test_number == 888008) {
            /*
            ./multigrid -test 888008 --3d --usecn --adaptiveH -cneps 0.00001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -bc 1 -mg_level 4 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours
            ./multigrid -test 888008 --3d --usecn -cneps 0.00001 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 --bcproject -bc 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 -o pn
            */
            sim.end_frame = 120;
            sim.dx = 0.01;
            sim.gravity = -9.81 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            int ppc = 8;

            if (1) {
                // Jello
                T nu = 0.4;
                T rho = 0.96e3;
                T E = 1e4;

                // Aluminum
                // T nu = 0.33;
                // T rho = 2e3;
                // T E = 1e8;
                // T yield = 2e6;

                // RandomNumber<T> random;
                // StdVector<TV> meshed_points;
                // readPositionObj("armadillos1.obj", meshed_points);
                // MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/armadillos1.vdb", rho, ppc);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFile("LevelSets/armadillos1.vdb", rho, ppc);

                CorotatedIsotropic<T, dim> model(E, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
                // VonMisesFixedCorotated<T, dim> p(yield);
                // particles_handle.addPlasticity(model, p, "F");
            }
            {
                // Jello
                T nu = 0.4;
                T rho = 1e3;
                T E = 1e5;

                // Aluminum
                // T nu = 0.33;
                // T rho = 2.7e3;
                // T E = 69e9;
                // T yield = 240e6;

                // RandomNumber<T> random;
                // StdVector<TV> meshed_points;
                // readPositionObj("armadillos2.obj", meshed_points);
                // MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/armadillos2.vdb", rho, ppc);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFile("LevelSets/armadillos2.vdb", rho, ppc);
                CorotatedIsotropic<T, dim> model(E, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
                // VonMisesFixedCorotated<T, dim> p(yield);
                // particles_handle.addPlasticity(model, p, "F");
            }

            auto ceilTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
                T t = time;
                T shrink_speed = 0.2;
                T shrink_time = 3.5;
                T hold_time =10;
                ZIRAN_ASSERT(1 - shrink_speed * shrink_time > 0);
                if (t < shrink_time) {
                    T theta = 0;
                    Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                    object.setRotation(rotation);
                    TV omega(0, 0, 0);
                    object.setAngularVelocity(omega);
                    TV translation_velocity(0, 0, 0);
                    TV translation(0.53, 0.45, 0.54);
                    object.setTranslation(translation, translation_velocity);
                    object.setScaling(1 - shrink_speed * t, -shrink_speed);}
                else if (t < hold_time) {
                    t = shrink_time;
                    T theta = 0;
                    Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                    object.setRotation(rotation);
                    TV omega(0, 0, 0);
                    object.setAngularVelocity(omega);
                    TV translation_velocity(0, 0, 0);
                    TV translation(0.53, 0.45, 0.54);} };

            TV center(0, 0, 0);
            Sphere<T, dim> sphere1(center, 0.4);
            Sphere<T, dim> sphere2(center, 0.15);
            DifferenceLevelSet<T, dim> materialRegionLevelSet;
            materialRegionLevelSet.add(sphere1, sphere2);
            AnalyticCollisionObject<T, dim> ceilObject(ceilTransform, materialRegionLevelSet, AnalyticCollisionObject<T, dim>::SLIP);
            init_helper.addAnalyticCollisionObject(ceilObject);
        }

        // car dragon
        if (test_number == 888999222) {
            // ./multigrid -test 888999222 --adaptiveH  --3d --usecn -cneps 1e-6 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours_car;
            // ./multigrid -test 888999222 --3d --usecn -cneps 1e-6 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 -o pn_car;

            sim.end_frame = 1800;

            sim.dx = 0.04 * 2;

            sim.gravity = -1 * TV::Unit(1);
            sim.gravity = TV();

            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt / 10.;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = std::sqrt(HOTSettings::characterNorm); // extended gast15

            //T nu = 0.4;
            //T rho_soft_rubber = 2e3;
            //T youngs_soft_rubber = 5e5;
            //T rho_wood = 2e3;
            //T youngs_wood = 1e8;
            T nu = 0.33;
            T rho_soft_rubber = 2.7e3;
            T youngs_soft_rubber = 69e9;
            T rho_wood = 2e3;
            T youngs_wood = 1e5;
            int ppc = 8;
            T yield = 240e6 * 1e-3;

            //sim.output_dir.path = destination("output/FCRtimingFractureTwist", sim.step.max_dt, sim.dx, youngs, ppc, sim.particles.count);

#if 0
#else
            // two cars
            {
                //StdVector<TV> meshed_points;
                //readPositionObj("twocars.obj", meshed_points);
                //MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/twocars.vdb", rho_soft_rubber, 8);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFile("LevelSets/twocars.vdb", rho_soft_rubber, 8);
                CorotatedIsotropic<T, dim> model(youngs_soft_rubber, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
                VonMisesFixedCorotated<T, dim> p(yield);
                particles_handle.addPlasticity(model, p, "F");
                auto initial_translation = [=](int index, Ref<T> mass, TV& X, TV& V) {
                    //std::cout << "particle " << index << "," << X << std::endl;
                    //std::cout << "particle " << index << "," << X(0) << std::endl;
                    if (X(0) < 3.2)
                        V = TV(2, 0, 0);
                    else
                        V = TV(0, 0, 0);

                    X(0) += 1;
                };
                particles_handle.transform(initial_translation);
            }
            {
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFile("LevelSets/dragon_bottom.vdb", rho_wood * 0.1, 8);
                CorotatedIsotropic<T, dim> model(youngs_wood, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
                auto initial_translation = [=](int index, Ref<T> mass, TV& X, TV& V) {
                    X(0) += 1;
                };
                particles_handle.transform(initial_translation);
            }
            { // floor
                TV ground_origin(0, 2.6 - sim.dx * 2, 0);
                TV ground_normal(0, 1, 0);
                HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
                AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(ground_object);
            }
            { // floor
                TV ground_origin(6.7, 0, 0);
                TV ground_normal(-1, 0, 0);
                HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
                AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(ground_object);
            }
#endif
        }

        if (test_number == 888999000) {
            // ./multigrid -test 888999000 --adaptiveH  --3d --usecn -cneps 1e-6 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours_donut_final;
            // ./multigrid -test 888999000 --3d --usecn -cneps 1e-6 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 -o pn_donut_final;

            sim.end_frame = 80;

            //sim.dx = 0.01 / 1.32;
            sim.dx = 0.01 / 1.32 * 0.75;
            //sim.dx = 0.02;

            sim.gravity = -1 * TV::Unit(1);

            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt / 10.;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = std::sqrt(HOTSettings::characterNorm); // extended gast15

            //T nu = 0.4;
            //T rho_soft_rubber = 2e3;
            //T youngs_soft_rubber = 5e5;
            //T rho_wood = 2e3;
            //T youngs_wood = 1e8;
            T nu = 0.33;
            T rho_soft_rubber = 2.7e3;
            T youngs_soft_rubber = 69e9;
            T rho_wood = 2e3;
            T youngs_wood = 1e5;
            int ppc = 8;
            T yield = 240e6;

            //sim.output_dir.path = destination("output/FCRtimingFractureTwist", sim.step.max_dt, sim.dx, youngs, ppc, sim.particles.count);

            //RandomNumber<T> random;

            // metal bar
            {
                StdVector<TV> meshed_points;
                readPositionObj("bar.obj", meshed_points);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/bar.vdb", rho_soft_rubber, ppc);
                CorotatedIsotropic<T, dim> model(youngs_soft_rubber, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
                VonMisesFixedCorotated<T, dim> p(yield);
                particles_handle.addPlasticity(model, p, "F");
            }
            ZIRAN_INFO("bar count: ", sim.particles.count);

            // elastic donut
            {

                T half = 0.04;
                RandomNumber<T> random;
                int count = 0;
                TV torus_center;
                T torus_radius;
                for (int i = -1; i <= -1; i += 2) {
                    for (int j = -1; j <= -1; j += 2) {
                        T dx = i * 0.085, dz = j * 0.085;
                        dx = 0;
                        dz = 0;
                        T ss[4] = { 0.25, 0.65, 0.55, 0.45 };
                        TV center(2 + dx, 2.8 + ss[count++] + 0.23, 2 + dz);
                        T radius = half * (2 + random.randReal(-0.2, 0.3));

                        Vector<T, 4> rotation(std::cos(M_PI / 2 / 2), std::sin(M_PI / 2 / 2), 0, 0);
                        TV trans = center;
                        Torus<T, dim> torus(radius, radius / 3, rotation, trans);
                        torus_center = center;
                        torus_radius = radius;
                        MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSetFullyRandom(torus, rho_wood, ppc);
                        //Sphere<T, dim> sphere(center, radius);
                        //MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSet(sphere, rho_wood, ppc);
                        CorotatedIsotropic<T, dim> model2(youngs_wood, nu);
                        model2.project = HOTSettings::project;
                        particles_handle2.addFBasedMpmForce(model2);
                    }
                }
            }
            //ZIRAN_INFO("bar+donut count: ", sim.particles.count);

            // elastic candy
            {
                //StdVector<TV> meshed_points;
                //readPositionObj("candy.obj", meshed_points);
                //MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/candy.vdb", rho_wood, ppc);
                //CorotatedIsotropic<T, dim> model2(youngs_wood, nu);
                //model2.project = HOTSettings::project;
                //particles_handle2.addFBasedMpmForce(model2);
            }
            ZIRAN_INFO("bar+donut+candy count: ", sim.particles.count);

            for (int c = 0; c < 2; c++) {
                T y = c ? 3.7 : 2.8;
                T s = c ? 1 : -1;
                T rise_speed = c ? 0 : 0;
                auto ceilTransform = [y, s, rise_speed](T time, AnalyticCollisionObject<T, dim>& object) {
                    T theta = s * (T)100.0 / 180 * M_PI; // 100 for final example
                    T t = time;
                    if (false) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, theta, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, rise_speed, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = 0;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                T theta = 0;
                Vector<T, 4> rotation(std::cos(theta / 2), std::sin(theta / 2), 0, 0);
                TV translation(0, 0, 0);
                CappedCylinder<T, dim> cylinder(/*radius*/ .2, /*height*/ .1, rotation, translation);
                AnalyticCollisionObject<T, dim> ceilObject(ceilTransform, cylinder, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(ceilObject);
            }

            // moving spheres
            {
                TV center(2.3, 3.54, 2);
                T radius = 0.1;
                Sphere<T, dim> sphere(center, radius);
                auto sphereTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
                    TV center(2.3, 3.54, 2);
                    T speed = -0.1;
                    T theta = 0;
                    T t = time;
                    T ttt = 52. / 24.;
                    if (t < ttt) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(theta, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(speed, 0, 0);
                        TV translation = TV(speed * t, 0, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = ttt * 100;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation = TV(speed * t, 0, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                AnalyticCollisionObject<T, dim> sphereObject(sphereTransform, sphere, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(sphereObject);
            }
            // moving spheres
            {
                TV center(1.7, 3., 2);
                T radius = 0.1;
                Sphere<T, dim> sphere(center, radius);
                auto sphereTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
                    TV center(1.7, 3., 2);
                    T speed = 0.1;
                    T theta = 0;
                    T t = time;
                    T ttt = 52. / 24.;
                    if (t < ttt) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(theta, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(speed, 0, 0);
                        TV translation = TV(speed * t, 0, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = ttt * 100;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation = TV(speed * t, 0, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                AnalyticCollisionObject<T, dim> sphereObject(sphereTransform, sphere, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(sphereObject);
            }

            //Set up a ground plane
            TV ground_origin(0, 2.75, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            ground_object.setFriction(1000.0);
            init_helper.addAnalyticCollisionObject(ground_object); // add ground
        }

        // elastic metal elastic
        if (test_number == 888999111) {
            // ./multigrid -test 888999111 --adaptiveH  --3d --usecn -cneps 1e-6 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours_box_final;
            // ./multigrid -test 888999111 --3d --usecn -cneps 1e-6 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 -o pn_box_final;

            sim.end_frame = 80;

            //sim.dx = 0.04 / 2.48;
            sim.dx = 0.02 / 2.48;

            //sim.gravity = -1 * TV::Unit(1);
            sim.gravity = TV::Zero();

            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt / 10.;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = std::sqrt(HOTSettings::characterNorm); // extended gast15

            //T nu = 0.4;
            //T rho_soft_rubber = 2e3;
            //T youngs_soft_rubber = 5e5;
            //T rho_wood = 2e3;
            //T youngs_wood = 1e8;
            T nu = 0.33;
            T rho_soft_rubber = 2.7e3;
            T youngs_soft_rubber = 69e9;
            T rho_wood = 2e3;
            T youngs_wood = 2e5;
            int ppc = 8;
            T yield = 240e6;

            //sim.output_dir.path = destination("output/FCRtimingFractureTwist", sim.step.max_dt, sim.dx, youngs, ppc, sim.particles.count);

#if 0
#else
            // left elastic box
            TV length(.5, .08, .5);
            TV center(2, 4, 2);
            TV center2 = center + TV(.5 - 0.04, 0, 0);
            TV center3 = center + TV(1. - 0.08, 0, 0);
            {
                AxisAlignedAnalyticBox<T, dim> box(center - length * .5, center + length * .5);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSetFullyRandom(box, rho_wood, ppc);
                CorotatedIsotropic<T, dim> model(youngs_wood, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
            }
            //// middle metal box
            //{
            //AxisAlignedAnalyticBox<T, dim> box(center2-length*.5,center2+length*.5);
            //MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(box, rho_soft_rubber, ppc);
            //CorotatedIsotropic<T, dim> model(youngs_soft_rubber, nu);
            //model.project = HOTSettings::project;
            //particles_handle.addFBasedMpmForce(model);
            //VonMisesFixedCorotated<T, dim> p(yield);
            //particles_handle.addPlasticity(model, p, "F");
            //}
            // obj for metal box
            {
                StdVector<TV> meshed_points;
                readPositionObj("metal_middle.obj", meshed_points);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/metal_middle.vdb", rho_soft_rubber, ppc);
                CorotatedIsotropic<T, dim> model(youngs_soft_rubber, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
                VonMisesFixedCorotated<T, dim> p(yield);
                particles_handle.addPlasticity(model, p, "F");
            }
            // right elastic box
            {
                AxisAlignedAnalyticBox<T, dim> box(center3 - length * .5, center3 + length * .5);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSetFullyRandom(box, rho_wood, ppc);
                CorotatedIsotropic<T, dim> model(youngs_wood, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
            }

#endif
            // moving spheres
            {
                T radius = 0.1;
                Sphere<T, dim> sphere(TV(2.5, 4.2, 2), radius);
                auto sphereTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
                    TV center = TV(2.5, 4.2, 2);
                    T speed = -0.3;
                    T theta = 0;
                    T t = time;
                    T ttt = 52. / 24.;
                    if (t < ttt) {
                        //Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        //object.setRotation(rotation);
                        //TV omega(theta, 0, 0);
                        //object.setAngularVelocity(omega);
                        TV translation_velocity(0, speed, 0);
                        TV translation = TV(0, speed * t, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = ttt * 100;
                        //Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        //object.setRotation(rotation);
                        //TV omega(0, 0, 0);
                        //object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation = TV(speed * t, 0, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                AnalyticCollisionObject<T, dim> sphereObject(sphereTransform, sphere, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(sphereObject);
            }

            // left wall
            {
                // 2-.5*.5+0.1
                TV wall_origin(1.85 + 0.1, 0, 0);
                TV wall_normal(1, 0, 0);
                HalfSpace<T, dim> wall_ls(wall_origin, wall_normal);
                AnalyticCollisionObject<T, dim> wall_object(wall_ls, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(wall_object);
            }
            // right wall
            {
                // 2+1-0.08+0.5*.5-0.1
                TV wall_origin(3.07 - 0.1, 0, 0);
                TV wall_normal(-1, 0, 0);
                HalfSpace<T, dim> wall_ls(wall_origin, wall_normal);
                AnalyticCollisionObject<T, dim> wall_object(wall_ls, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(wall_object);
            }
            //Set up a ground plane
            {
                TV ground_origin(0, 3.2, 0);
                TV ground_normal(0, 1, 0);
                HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
                AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
                ground_object.setFriction(1000.0);
                init_helper.addAnalyticCollisionObject(ground_object); // add ground
            }
        }

        if (test_number == 888999) {

            // ./multigrid -test 888999 --adaptiveH  --3d --usecn -cneps 1e-6 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours_smalldt_easier_middle_res;
            // ./multigrid -test 888999 --3d --usecn -cneps 1e-6 -lsolver 2 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 1 -mg_scale 0 -mg_omega 1.0 -coarseSolver 0 -smoother 0 -dbg 0 -o pn_smalldt_easier_middle_res;

            sim.end_frame = 80;

            sim.dx = 0.01 / 1.32;

            sim.gravity = -1 * TV::Unit(1);

            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt / 10.;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = std::sqrt(HOTSettings::characterNorm); // extended gast15

            //T nu = 0.4;
            //T rho_soft_rubber = 2e3;
            //T youngs_soft_rubber = 5e5;
            //T rho_wood = 2e3;
            //T youngs_wood = 1e8;
            T nu = 0.33;
            T rho_soft_rubber = 2.7e3;
            T youngs_soft_rubber = 69e9;
            T rho_wood = 2e3;
            T youngs_wood = 1e5;
            int ppc = 8;
            T yield = 240e6;

            //sim.output_dir.path = destination("output/FCRtimingFractureTwist", sim.step.max_dt, sim.dx, youngs, ppc, sim.particles.count);

            T half = 0.04;
            RandomNumber<T> random;
            int count = 0;
            TV torus_center;
            T torus_radius;
            for (int i = -1; i <= -1; i += 2) {
                for (int j = -1; j <= -1; j += 2) {
                    T dx = i * 0.085, dz = j * 0.085;
                    dx = 0;
                    dz = 0;
                    T ss[4] = { 0.25, 0.65, 0.55, 0.45 };
                    TV center(2 + dx, 2.8 + ss[count++] + 0.23, 2 + dz);
                    T radius = half * (2 + random.randReal(-0.2, 0.3));

                    Vector<T, 4> rotation(std::cos(M_PI / 2 / 2), std::sin(M_PI / 2 / 2), 0, 0);
                    TV trans = center;
                    Torus<T, dim> torus(radius, radius / 3, rotation, trans);
                    torus_center = center;
                    torus_radius = radius;
                    MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSetFullyRandom(torus, rho_wood, ppc);
                    //Sphere<T, dim> sphere(center, radius);
                    //MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSet(sphere, rho_wood, ppc);
                    CorotatedIsotropic<T, dim> model2(youngs_wood, nu);
                    model2.project = HOTSettings::project;
                    particles_handle2.addFBasedMpmForce(model2);
                }
            }
            ZIRAN_INFO("torus count: ", sim.particles.count);
            for (int i = -1; i <= -1; i += 2) {
                for (int j = -1; j <= -1; j += 2) {
                    T dx = i * 0.085, dz = j * 0.085;
                    dx = 0;
                    dz = 0;
                    AxisAlignedAnalyticBox<T, dim> box(TV(2 - half, 2.8, 2 - half) + TV(dx, 0, dz), TV(2 + half, 3.7, 2 + half) + TV(dx, 0, dz));
                    Sphere<T, dim> sphere(torus_center, torus_radius * 0.9);
                    DifferenceLevelSet<T, dim> materialRegionLevelSet;
                    materialRegionLevelSet.add(box, sphere);
                    MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(materialRegionLevelSet, rho_soft_rubber, ppc);
                    CorotatedIsotropic<T, dim> model(youngs_soft_rubber, nu);
                    model.project = HOTSettings::project;
                    particles_handle.addFBasedMpmForce(model);
                    VonMisesFixedCorotated<T, dim> p(yield);
                    particles_handle.addPlasticity(model, p, "F");
                }
            }

            for (int c = 0; c < 2; c++) {
                T y = c ? 3.7 : 2.8;
                T s = c ? 1 : -1;
                T rise_speed = c ? 0 : 0;
                auto ceilTransform = [y, s, rise_speed](T time, AnalyticCollisionObject<T, dim>& object) {
                    T theta = s * (T)100.0 / 180 * M_PI; // 100 for final example
                    T t = time;
                    if (false) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, theta, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, rise_speed, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = 0;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                T theta = 0;
                Vector<T, 4> rotation(std::cos(theta / 2), std::sin(theta / 2), 0, 0);
                TV translation(0, 0, 0);
                CappedCylinder<T, dim> cylinder(/*radius*/ .2, /*height*/ .1, rotation, translation);
                AnalyticCollisionObject<T, dim> ceilObject(ceilTransform, cylinder, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(ceilObject);
            }

            // moving spheres
            {
                TV center(2.3, 3.54, 2);
                T radius = 0.1;
                Sphere<T, dim> sphere(center, radius);
                auto sphereTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
                    TV center(2.3, 3.54, 2);
                    T speed = -0.1;
                    T theta = 0;
                    T t = time;
                    T ttt = 52. / 24.;
                    if (t < ttt) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(theta, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(speed, 0, 0);
                        TV translation = TV(speed * t, 0, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = ttt * 100;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation = TV(speed * t, 0, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                AnalyticCollisionObject<T, dim> sphereObject(sphereTransform, sphere, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(sphereObject);
            }
            // moving spheres
            {
                TV center(1.7, 3., 2);
                T radius = 0.1;
                Sphere<T, dim> sphere(center, radius);
                auto sphereTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
                    TV center(1.7, 3., 2);
                    T speed = 0.1;
                    T theta = 0;
                    T t = time;
                    T ttt = 52. / 24.;
                    if (t < ttt) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(theta, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(speed, 0, 0);
                        TV translation = TV(speed * t, 0, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = ttt * 100;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation = TV(speed * t, 0, 0);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                AnalyticCollisionObject<T, dim> sphereObject(sphereTransform, sphere, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(sphereObject);
            }

            //Set up a ground plane
            TV ground_origin(0, 2.75, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            ground_object.setFriction(1000.0);
            init_helper.addAnalyticCollisionObject(ground_object); // add ground
        }

        if (test_number == 777009) {
            sim.end_frame = 13;
            sim.dx = 0.01;
            sim.gravity = -1 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            int ppc = 8;
            RandomNumber<T> random;
            TV domain_min(1.5, 1.5, 1.5);
            T h = 0.1;
            T thick = 0.05;
            T min_rho = 500;
            T max_rho = 4000;
            T min_E = 10000;
            T max_E = 10000000;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    TV min_corner = domain_min + TV(i * h, j * h, 0);
                    TV max_corner = min_corner + TV(h, h, thick);
                    AxisAlignedAnalyticBox<T, dim> box(min_corner, max_corner);
                    MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(box, random.randReal(min_rho, max_rho), ppc);
                    CorotatedIsotropic<T, dim> model(random.randReal(min_E, max_E), 0.38);
                    model.project = HOTSettings::project;
                    particles_handle.addFBasedMpmForce(model);
                }
            }

            T rise_speed = 1;
            auto ceilTransform = [rise_speed](T time, AnalyticCollisionObject<T, dim>& object) {
                T t = time;
                if (t < 1.6667) {
                    TV translation_velocity(0, rise_speed, 0);
                    TV translation(0, rise_speed * t, 0);
                    object.setTranslation(translation, translation_velocity);
                }
                else {
                    t = 1.6667;
                    TV translation_velocity(0, 0, 0);
                    TV translation(0, rise_speed * t, 0);
                    object.setTranslation(translation, translation_velocity);
                }
            };

            TV ceil_origin(0, 2 - h, 0);
            TV ceil_normal(0, -1, 0);
            HalfSpace<T, dim> ceil_ls(ceil_origin, ceil_normal);
            AnalyticCollisionObject<T, dim> ceil_object(ceilTransform, ceil_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ceil_object);

            TV ground_origin(0, 1.5 + h, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);
        }

        if (test_number == 777010) { // chain
            sim.end_frame = 240;
            sim.dx = 0.01;
            sim.gravity = -0 * TV::Unit(1);

            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = std::sqrt(HOTSettings::characterNorm); // extended gast15

            T nu = 0.4;
            T rho_soft_rubber = 2e3;
            T youngs_soft_rubber = 5e5;
            T rho_wood = 2e3;
            T youngs_wood = 3e9; //1e8;
            int ppc = 8;

            //sim.output_dir.path = destination("output/FCRtimingFractureTwist", sim.step.max_dt, sim.dx, youngs, ppc, sim.particles.count);
            T rbig = 0.2, rsmall = 0.04;
            T dh = 0.54;
            for (int i = 0; i < 4; i++) {
                Vector<T, 4> rotation(std::cos(0 / 2), 0, std::sin(0 / 2), 0);
                TV trans(1.3 + i * dh, 1.3, 1.3);
                Torus<T, dim> torus(rbig, rsmall, rotation, trans);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(torus, rho_soft_rubber, ppc);
                CorotatedIsotropic<T, dim> model(youngs_soft_rubber, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
            }
            for (int i = 0; i < 3; i++) {
                Vector<T, 4> rotation(std::cos(M_PI * .5 / 2), std::sin(M_PI * .5 / 2), 0, 0);
                TV trans(1.3 + dh / 2 + i * dh, 1.3, 1.3);
                Torus<T, dim> torus(rbig, rsmall, rotation, trans);
                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(torus, rho_wood, ppc);
                CorotatedIsotropic<T, dim> model(youngs_wood, nu);
                model.project = HOTSettings::project;
                particles_handle.addFBasedMpmForce(model);
            }

            for (int c = 0; c < 2; c++) {
                T shiftx = c ? (1.3 + dh / 2 + 3 * dh) : (1.3 + dh / 2 - dh);
                T s = c ? 1 : -1;
                T move_speed = c ? 0.2 : -0.2;
                T move_time = 2;
                T stop_time = 8;
                auto ceilTransform = [shiftx, s, move_speed, move_time, stop_time](T time, AnalyticCollisionObject<T, dim>& object) {
                    T theta = s * (T)250.0 / 180 * M_PI;
                    T t = time;
                    if (t < move_time) {
                        TV translation_velocity(move_speed, 0, 0);
                        TV translation(shiftx + move_speed * t, 1.3, 1.3);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else if (t < stop_time) {
                        t -= move_time;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(theta, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation(shiftx + move_speed * move_time, 1.3, 1.3);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = stop_time - move_time;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation(shiftx + move_speed * move_time, 1.3, 1.3);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                Vector<T, 4> rotation(std::cos(M_PI * .5 / 2), std::sin(M_PI * .5 / 2), 0, 0);
                TV trans(0, 0, 0);
                Torus<T, dim> torus(rbig, rsmall, rotation, trans);
                AnalyticCollisionObject<T, dim> torus_object(ceilTransform, torus, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(torus_object);
            }
        }

        if (test_number == 777011) { // lion (faceless)
            /* 
            ./multigrid -test 777011 --3d --usecn -cneps 0.00001 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o pn
            ./multigrid -test 777011 --3d --usecn -cneps 0.00001 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours
              */
            sim.end_frame = 80;
            sim.dx = 0.01;
            sim.gravity = -.3 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = std::sqrt(HOTSettings::characterNorm); // extended gast15

            // ground is at 0
            TV ground_origin(0, 0.6, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground([&](T, AnalyticCollisionObject<T, dim>&) {}, ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground);

            T rho = 2000;
            T ppc = 20;
            T E = 50000;

            // Embedding a mesh for rendering.
            StdVector<TV> meshed_points;
            readPositionObj("lion.obj", meshed_points);
            MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/lion.vdb", rho, ppc);
            auto initial_translation = [=](int index, Ref<T> mass, TV& X, TV& V) {
                X = X + TV(5, 5, 5);
            };
            particles_handle.transform(initial_translation);
            CorotatedIsotropic<T, dim> model(E, .3);
            model.project = HOTSettings::project;
            particles_handle.addFBasedMpmForce(model);

            TV translation_freeze = TV(5, 5.345, 5) + TV(0, 24 * 0.0015, 0) * 40. / 24.;
            T rotation_freeze = (T)24 * 5 / 180 * M_PI * 40. / 24.;

            auto cylinder_transform = [translation_freeze, rotation_freeze](T time, AnalyticCollisionObject<T, dim>& object) {
                T frame_dt = 1. / 24.;
                T frame_move = 40;
                T frame_freeze = 50;
                if (time < frame_move * frame_dt) {
                    // rotate
                    T theta = (T)24 * 5 / 180 * M_PI;
                    Vector<T, 4> rotation(std::cos(theta * time / 2), 0, std::sin(theta * time / 2), 0);
                    object.setRotation(rotation);
                    TV omega(0, theta, 0);
                    object.setAngularVelocity(omega);

                    TV translation_fixed(5, 5.345, 5);

                    TV translation_velocity(0, 24 * 0.0015, 0);
                    TV translation = translation_fixed + translation_velocity * time;
                    object.setTranslation(translation, translation_velocity);
                }
                else if (time < frame_freeze * frame_dt) {

                    Vector<T, 4> rotation(std::cos(rotation_freeze / 2), 0, std::sin(rotation_freeze / 2), 0);
                    object.setRotation(rotation);
                    TV omega(0, 0, 0);
                    object.setAngularVelocity(omega);

                    TV translation = translation_freeze;
                    TV translation_velocity(0, 0, 0);
                    object.setTranslation(translation, translation_velocity);
                }
                else {
                    TV translation(0, 0, 0);
                    TV translation_velocity(0, 0, 0);
                    object.setTranslation(translation, translation_velocity);
                }
            };

            T radius = .09;
            T height = .2;
            T theta = 0;
            Vector<T, 4> rotation(std::cos(theta / 2), std::sin(theta / 2), 0, 0);
            TV translation(0, 0, 0);
            CappedCylinder<T, dim> cylinder(radius, height, rotation, translation);
            AnalyticCollisionObject<T, dim> sphere_object1(cylinder_transform, cylinder, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(sphere_object1);

            TV box_center(5, 5, 5);
            TV box_size(.2, .1, 2);
            TV box_min_corner = box_center - box_size * .5;
            TV box_max_corner = box_center + box_size * .5;
            AxisAlignedAnalyticBox<T, dim> box(box_min_corner, box_max_corner);
            AnalyticCollisionObject<T, dim> box_object(box, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(box_object);
        }

        if (test_number == 777012) { // a glass of food
            /*
            ./multigrid -test 777012 --3d --usecn -cneps 0.000001 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o pn
            ./multigrid -test 777012 --3d --usecn -cneps 0.000001 --l2norm  -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours
            */
            sim.end_frame = 96;
            sim.dx = 0.01;
            sim.gravity = -9.81 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            int ppc = 8;

            sim.end_time_step_callbacks.push_back(
                [this, ppc](int frame, int substep) {
                    if (frame <= 72 && frame % 6 == 0 && substep == 1) {
                        RandomNumber<T> r(frame);
                        TM R;
                        r.randRotation(R);
                        Eigen::Quaternion<T> q(R);
                        Vector<T, 4> rotation(q.w(), q.x(), q.y(), q.z());
                        TV trans = TV(3.45, 2.926, 3.04) + TV(r.randReal(-0.1, 0.1), 0, r.randReal(-0.1, 0.1));
                        Torus<T, dim> torus(.08, .08 / 3, rotation, trans);
                        // Sphere<T, dim> sphere(TV(3.45, 3.1, 3.04), .05);

                        int id = frame / 6; // id=1,2,3,4,...12
                        if (id < 5 || id % 2 == 1) { // odd id
                            T rho = 1000, E = 1e5;
                            MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(torus, rho, ppc);
                            CorotatedIsotropic<T, dim> model(E, .35);
                            SnowPlasticity<T> p(0, 0.01, 0.01, 0.0001, 1000);
                            ph.addPlasticity(model, p, "F");
                            model.project = HOTSettings::project;
                            ph.addFBasedMpmForce(model);
                        }
                        else {
                            T rho = 2.7e3, E = 69e9;
                            MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(torus, rho, ppc);
                            CorotatedIsotropic<T, dim> model(E, .35);
                            model.project = HOTSettings::project;
                            // VonMisesFixedCorotated<T, dim> p(240e6);
                            // ph.addPlasticity(model, p, "F");
                            ph.addFBasedMpmForce(model);
                        }
                    }
                });

            TV ground_origin(0, 2.57, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);

            // CappedCylinder<T, dim> cylinder1(0.33, 1, Vector<T, 4>(1, 0, 0, 0), TV(3.45, 2.71, 3.04));
            // CappedCylinder<T, dim> cylinder2(0.27, 1, Vector<T, 4>(1, 0, 0, 0), TV(3.45, 3.07, 3.04));
            // DifferenceLevelSet<T, dim> materialRegionLevelSet;
            // materialRegionLevelSet.add(cylinder1, cylinder2);
            // AnalyticCollisionObject<T, dim> bowl(materialRegionLevelSet, AnalyticCollisionObject<T, dim>::STICKY);
            // init_helper.addAnalyticCollisionObject(bowl);
        }

        if (test_number == 777013) { // pumpkin and nail
            /*
            ./multigrid -test 777013 --3d --usecn -cneps 0.000001 -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o pn
            ./multigrid -test 777013 --3d --usecn -cneps 0.000001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 --adaptiveH -o oursH
            */
            sim.end_frame = 110;
            sim.dx = 0.01;
            sim.gravity = 0 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt / 2;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            TV shift(1.1, 1.1, 1.1);

            T pumpkin_rho = 1000, pumpkin_E = 2e4;
            T base_rho = 1000, base_E = 3e4;
            T nail_rho = 2700, nail_E = 69e9, nail_yield = 240e6;

            {
                MpmParticleHandleBase<T, dim> ph = init_helper.sampleFromVdbFile("LevelSets/pumpkin.vdb", pumpkin_rho, 8);
                ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
                CorotatedIsotropic<T, dim> model(pumpkin_E, .3);
                ph.addFBasedMpmForce(model);
                ZIRAN_INFO("pumpkin: [", ph.particle_range.lower, ",", ph.particle_range.upper, ")");
            }

            {
                CappedCylinder<T, dim> cc(0.264, 0.176, Vector<T, 4>(1, 0, 0, 0), TV(0, -0.09, 0));
                MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(cc, base_rho, 8);
                ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
                CorotatedIsotropic<T, dim> model(base_E, .3);
                ph.addFBasedMpmForce(model);
                ZIRAN_INFO("base: [", ph.particle_range.lower, ",", ph.particle_range.upper, ")");
            }

            {
                MpmParticleHandleBase<T, dim> ph = init_helper.sampleFromVdbFile("LevelSets/nail.vdb", nail_rho, 8);
                ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
                CorotatedIsotropic<T, dim> model(nail_E, .3);
                ph.addFBasedMpmForce(model);
                VonMisesFixedCorotated<T, dim> p(nail_yield);
                ph.addPlasticity(model, p, "F");
                ZIRAN_INFO("nails: [", ph.particle_range.lower, ",", ph.particle_range.upper, ")");
            }

            HalfSpace<T, dim> ground_ls(TV(0, -0.15, 0) + shift, TV(0, 1, 0));
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);

            auto hammerTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
                T v = -0.2;
                T stop = 2.8;
                T t = time;
                if (t < stop)
                    object.setTranslation(TV(0, v * t, 0), TV(0, v, 0));
                else
                    object.setTranslation(TV(5, v * stop, 5), TV(0, 0, 0));
            };
            Sphere<T, dim> hammer(TV(.1, .99, 0) + shift, .06);
            AnalyticCollisionObject<T, dim> ohammer(hammerTransform, hammer, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ohammer);
        }

        if (test_number == 777014) { // fracture
            /*
	    ./multigrid -test 777014 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o fracture_pn
	    ./multigrid -test 777014 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --matfree -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o fracture_pnfree
	    ./multigrid -test 777014 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o fracture_ours
        ./multigrid -test 777014 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 10000 -mg_scale 0 -coarseSolver 2 -smoother 2 -dbg 0 -o fracture_LBFGS_H
        ./multigrid -test 777014 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o fracture_MG_PNPCG
	  */

            sim.end_frame = 131;

            sim.dx = 0.01; // 0.005 for final example

            sim.gravity = -9.8 * TV::Unit(1);

            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt;

            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = 1e7; ///< use absolute tolerance

            //T rho = 2;
            //T nu = 0.4, youngs = 500;
            T nu = 0.4;
            T rho_soft_rubber = 2e3;
            T youngs_soft_rubber = 5e5;
            T rho_wood = 2e3;
            T youngs_wood = 5e9;
            int ppc = 8;

            sim.output_dir.path = destination("output/FCRtimingFractureTwist", sim.step.max_dt, sim.dx, youngs_soft_rubber, ppc, sim.particles.count);

            T half = 0.04;
            RandomNumber<T> random;
            for (int i = -1; i <= 1; i += 2) {
                for (int j = -1; j <= 1; j += 2) {
                    T dx = i * 0.085, dz = j * 0.085;
                    AxisAlignedAnalyticBox<T, dim> box(TV(2 - half, 2.8, 2 - half) + TV(dx, 0, dz), TV(2 + half, 3.7, 2 + half) + TV(dx, 0, dz));
                    MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(box, rho_soft_rubber, ppc);
                    CorotatedIsotropic<T, dim> model(youngs_soft_rubber, nu);
                    model.project = HOTSettings::project;
                    particles_handle.addFBasedMpmForce(model);
                }
            }
            ZIRAN_INFO("pillar count: ", sim.particles.count);
            int count = 0;
            for (int i = -1; i <= 1; i += 2) {
                for (int j = -1; j <= 1; j += 2) {
                    T dx = i * 0.085, dz = j * 0.085;
                    T ss[4] = { 0.25, 0.65, 0.55, 0.45 };
                    TV center(2 + dx, 2.8 + ss[count++], 2 + dz);
                    T radius = half * (2 + random.randReal(-0.2, 0.3));
                    Sphere<T, dim> sphere(center, radius);
                    MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSet(sphere, rho_wood, ppc);
                    CorotatedIsotropic<T, dim> model2(youngs_wood, nu);
                    model2.project = HOTSettings::project;
                    particles_handle2.addFBasedMpmForce(model2);
                }
            }

            // auto ptrans = [](int index, Ref<T> mass, TV& X, TV& V) { X += TV(0, 0, 0); };
            // particles_handle.transform(ptrans);

            for (int c = 0; c < 2; c++) {
                T y = c ? 3.7 : 2.8;
                T s = c ? 1 : -1;
                T rise_speed = c ? 0.1 : 0; // 0.1 for final example
                auto ceilTransform = [y, s, rise_speed](T time, AnalyticCollisionObject<T, dim>& object) {
                    T theta = s * (T)100.0 / 180 * M_PI; // 100 for final example
                    T t = time;
                    if (t < 3) {
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, theta, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, rise_speed, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                    else {
                        t = 3;
                        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
                        object.setRotation(rotation);
                        TV omega(0, 0, 0);
                        object.setAngularVelocity(omega);
                        TV translation_velocity(0, 0, 0);
                        TV translation(2, y + rise_speed * t, 2);
                        object.setTranslation(translation, translation_velocity);
                    }
                };
                T theta = 0;
                Vector<T, 4> rotation(std::cos(theta / 2), std::sin(theta / 2), 0, 0);
                TV translation(0, 0, 0);
                CappedCylinder<T, dim> cylinder(/*radius*/ .2, /*height*/ .1, rotation, translation);
                AnalyticCollisionObject<T, dim> ceilObject(ceilTransform, cylinder, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(ceilObject);
            }

            //Set up a ground plane
            TV ground_origin(0, 2.75, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            ground_object.setFriction(1000.0);
            init_helper.addAnalyticCollisionObject(ground_object); // add ground
        }

        if (test_number == 777015) { // truss
            /*  
	    ./multigrid -test 777015 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o truss_ours
	    ./multigrid -test 777015 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o truss_pn
	    ./multigrid -test 777015 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --matfree -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o truss_pnfree
        */
            sim.end_frame = 100;
            sim.dx = 0.015;
            sim.gravity = -9.8 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            TV shift(1.1, 1.1, 1.1);

            // Aluminum
            T nu = 0.33;
            T rho = 2.7e3;
            T E = 69e9;
            T yield = 240e6;

            // tissue
            T nu2 = 0.4;
            T rho2 = 2.7e3;
            T E2 = 1e5;

            {
                StdVector<TV> meshed_points0;
                readPositionObj("truss.obj", meshed_points0);
                MpmParticleHandleBase<T, dim> ph = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points0, "LevelSets/truss.vdb", rho, 12);
                CorotatedIsotropic<T, dim> model(E, nu);
                model.project = HOTSettings::project;
                ph.addFBasedMpmForce(model);
                VonMisesFixedCorotated<T, dim> p(yield);
                ph.addPlasticity(model, p, "F");
                ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
            }

            // {
            //     Sphere<T, dim> sphere(TV(0, 0.51, 0), 0.418);
            //     MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSet(sphere, rho2, 6);
            //     CorotatedIsotropic<T, dim> model2(E2, nu2);
            //     model2.project = HOTSettings::project;
            //     particles_handle2.addFBasedMpmForce(model2);
            //     particles_handle2.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
            // }

            T speed = -0.2;
            auto ceilTransform = [speed](T time, AnalyticCollisionObject<T, dim>& object) {
                T t = time;
                if (t < 4) {
                    TV translation_velocity(0, speed, 0);
                    TV translation(0, speed * t, 0);
                    object.setTranslation(translation, translation_velocity);
                }
                else {
                    t = 4;
                    TV translation_velocity(0, 0, 0);
                    TV translation(0, speed * t, 0);
                    object.setTranslation(translation, translation_velocity);
                }
            };
            TV ceil_origin = TV(0, 1.05, 0) + shift;
            TV ceil_normal(0, -1, 0);
            HalfSpace<T, dim> ceil_ls(ceil_origin, ceil_normal);
            AnalyticCollisionObject<T, dim> ceil_object(ceilTransform, ceil_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ceil_object);

            TV ground_origin = TV(0, 0, 0) + shift;
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);

            sim.end_frame_callbacks.emplace_back([&](int frame) {
                std::string filename = sim.output_dir.absolutePath(sim.outputFileName("stress", ".bgeo"));
                Partio::ParticlesDataMutable* parts = Partio::create();
                Partio::ParticleAttribute posH, kirchhoffH, vonmisesstressH;
                posH = parts->addAttribute("position", Partio::VECTOR, 3);
                kirchhoffH = parts->addAttribute("kirchhoffStress", Partio::VECTOR, 1);
                vonmisesstressH = parts->addAttribute("vonMisesStress", Partio::VECTOR, 1);
                using TCONST = CorotatedIsotropic<T, dim>;
                using Scratch = typename TCONST::Scratch;
                AttributeName<TCONST> model_name(TCONST::name());
                AttributeName<Scratch> scratch_name(TCONST::scratch_name());
                auto& particles = sim.particles;
                StdVector<T> kirchhoffData(particles.count, T(1));
                StdVector<T> vonmisesstressData(particles.count, T(1));
                for (auto iter = particles.iter(model_name, scratch_name, F_name<T, dim>()); iter; ++iter) {
                    auto& model = iter.template get<0>();
                    auto& model_scratch = iter.template get<1>();
                    auto F = iter.template get<2>();
                    model.updateScratch(F, model_scratch);
                    TV sigma = model_scratch.sigma;
                    TV tau;
                    for (int d = 0; d < dim; d++) tau(d) = 2 * model.mu * (sigma(d) - 1) * sigma(d) + model.lambda * (model_scratch.J - 1) * model_scratch.J;
                    kirchhoffData[iter.entryId()] = tau.norm();
                    T trace_tau = tau.sum();
                    TV s = tau - TV::Ones() * (trace_tau / (T)dim);
                    vonmisesstressData[iter.entryId()] = s.norm();
                }
                for (int k = 0; k < particles.count; k++) {
                    int idx = parts->addParticle();
                    float* posP = parts->dataWrite<float>(posH, idx);
                    float* kirchhoffP = parts->dataWrite<float>(kirchhoffH, idx);
                    float* vonmisesstressP = parts->dataWrite<float>(vonmisesstressH, idx);
                    for (int d = 0; d < 3; ++d)
                        posP[d] = 0;
                    for (int d = 0; d < dim; ++d)
                        posP[d] = (float)particles.X.array[k](d);
                    kirchhoffP[0] = (float)kirchhoffData[k];
                    vonmisesstressP[0] = (float)vonmisesstressData[k];
                }
                Partio::write(filename.c_str(), *parts);
                parts->release();
            });
        }

        if (test_number == 777016) { // goo
            /*  
	    ./multigrid -test 777016 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o goo_ours
	    ./multigrid -test 777016 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o goo_pn
	    ./multigrid -test 777016 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --matfree -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o goo_pnfree
        */
            sim.end_frame = 100;
            sim.dx = 0.015;
            sim.gravity = -9.8 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            TV shift(1.1, 1.1, 1.1);

            T nu = 0.33;
            T rho = 1e3;
            T E = 1e7;
            T yield = 1;
            int ppc = 16;

            // create source collision object
            TV source_min_corner = TV(-0.02, -0.05, -0.02) + shift;
            TV source_max_corner = TV(0.02, 0.05, 0.02) + shift;
            AxisAlignedAnalyticBox<T, dim> source_box_ls(source_min_corner, source_max_corner);
            TV material_speed(0.0, -0.5, 0);
            SourceCollisionObject<T, dim> box_source(source_box_ls, material_speed);

            int source_id = init_helper.addSourceCollisionObject(box_source);
            init_helper.sampleSourceAtTheBeginning(source_id, rho, ppc);

            sim.end_time_step_callbacks.push_back(
                [this, source_id, rho, ppc, E, nu, yield](int frame, int substep) {
                    if (frame < 120) {
                        // add more particles from source Collision object
                        int N = init_helper.sourceSampleAndPrune(source_id, rho, ppc);
                        if (N) {
                            MpmParticleHandleBase<T, dim> source_particles_handle = init_helper.getParticlesFromSource(source_id, rho, ppc);
                            CorotatedIsotropic<T, dim> model(E, nu);
                            model.project = HOTSettings::project;
                            source_particles_handle.addFBasedMpmForce(model);

                            SnowPlasticity<T> p(0, 0.01, 0.0001, -2, 5);
                            source_particles_handle.addPlasticity(model, p, "F");

                            // VonMisesFixedCorotated<T, dim> p(yield);
                            // source_particles_handle.addPlasticity(model, p, "F");
                            T a = yield; // on Mac every variable in the [] must be used...
                        }
                    }
                });

            TV ground_origin = TV(0, -0.6, 0) + shift;
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);
        }

        if (test_number == 777017) { // boards
            /*  
	    ./multigrid -test 777017 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o boards_ours
	    ./multigrid -test 777017 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o boards_pn
	    ./multigrid -test 777017 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --matfree -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o boards_pnfree
        */
            sim.end_frame = 144;
            sim.dx = 0.007;
            sim.gravity = -9.8 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 1;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            // Aluminum
            T nu = 0.33;
            // T rho = 2.7e3;
            // T E = 69e9;
            // T yield = 240e6;

            TV shift(1.1, 1.1, 1.1);

            T cdx = 0.32, dy = 0.03, dz = 0.2;

            T rho = 1000, E = 1e5;

            // TV source_min_corner = TV(.4, 1, 0.18)+shift;
            // TV source_max_corner = TV( 0.45,  1.1,  0.22) + shift;
            // AxisAlignedAnalyticBox<T, dim> source_box_ls(source_min_corner, source_max_corner);
            Sphere<T, dim> source_sphere_ls(TV(.425, 1.05, 0.2) + shift, 0.03);

            TV material_speed(0.0, -0.5, 0);
            SourceCollisionObject<T, dim> box_source(source_sphere_ls, material_speed);
            int source_id = init_helper.addSourceCollisionObject(box_source);
            init_helper.sampleSourceAtTheBeginning(source_id, rho, 16);
            sim.end_time_step_callbacks.push_back(
                [this, source_id, rho, E](int frame, int substep) {
                    if (frame < 100) {
                        // add more particles from source Collision object
                        int N = init_helper.sourceSampleAndPrune(source_id, rho, 16);
                        if (N) {
                            MpmParticleHandleBase<T, dim> source_particles_handle = init_helper.getParticlesFromSource(source_id, rho, 16);
                            CorotatedIsotropic<T, dim> model(E, 0.3);
                            model.project = HOTSettings::project;
                            source_particles_handle.addFBasedMpmForce(model);

                            SnowPlasticity<T> p(0, 0.01, 0.001, -2, 5);
                            source_particles_handle.addPlasticity(model, p, "F");
                        }
                    }
                });
            {
                Vector<T, 4> rotation(1, 0, 0, 0);
                TV trans = TV(0.479, 0.612, 0.2) + shift;
                Torus<T, dim> torus(.08, .02, rotation, trans);
                T rho = 1000, E = 1e8;
                MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(torus, rho, 8);
                CorotatedIsotropic<T, dim> model(E, .35);
                model.project = HOTSettings::project;
                ph.addFBasedMpmForce(model);
            }

            // {AxisAlignedAnalyticBox<T, dim> cube(TV(0.35,1,0.1+dz/2), TV(0.45,1.1,0.1+dz/2+0.1)); //rectangular level set
            // MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(cube, rho, 8);
            // CorotatedIsotropic<T, dim> model(E, nu);
            // model.project = HOTSettings::project;
            // ph.addFBasedMpmForce(model);
            // ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; V=TV(0,-0.4,0);});
            // VonMisesFixedCorotated<T, dim> p(yield);
            // ph.addPlasticity(model, p, "F");}

            {
                AxisAlignedAnalyticBox<T, dim> cube(TV(0.15, 0.8, 0.1), TV(0.15 + cdx + 0.1, 0.8 + dy, 0.1 + dz)); //rectangular level set
                MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(cube, rho, 8);
                CorotatedIsotropic<T, dim> model(3e6, nu);
                model.project = HOTSettings::project;
                ph.addFBasedMpmForce(model);
                ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
                // VonMisesFixedCorotated<T, dim> p(yield);
                // ph.addPlasticity(model, p, "F");
            }

            {
                AxisAlignedAnalyticBox<T, dim> cube(TV(0.85 - cdx, 0.6, 0.1), TV(0.85, 0.6 + dy, 0.1 + dz)); //rectangular level set
                MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(cube, rho, 8);
                CorotatedIsotropic<T, dim> model(2e6, nu);
                model.project = HOTSettings::project;
                ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
                ph.addFBasedMpmForce(model);
                // VonMisesFixedCorotated<T, dim> p(yield);
                // ph.addPlasticity(model, p, "F");
            }

            {
                AxisAlignedAnalyticBox<T, dim> cube(TV(0.15, 0.23, 0.1), TV(0.85, 0.23 + dy, 0.1 + dz)); //rectangular level set
                MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(cube, rho, 8);
                CorotatedIsotropic<T, dim> model(2e5, nu);
                model.project = HOTSettings::project;
                ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
                ph.addFBasedMpmForce(model);
                // VonMisesFixedCorotated<T, dim> p(yield);
                // ph.addPlasticity(model, p, "F");
            }

            TV wallXmin_origin = TV(0.2, 0, 0) + shift;
            TV wallXmin_normal(1, 0, 0);
            HalfSpace<T, dim> wallXmin_ls(wallXmin_origin, wallXmin_normal);
            AnalyticCollisionObject<T, dim> wallXmin_object(wallXmin_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(wallXmin_object);

            TV wallXmax_origin = TV(0.8, 0, 0) + shift;
            TV wallXmax_normal(-1, 0, 0);
            HalfSpace<T, dim> wallXmax_ls(wallXmax_origin, wallXmax_normal);
            AnalyticCollisionObject<T, dim> wallXmax_object(wallXmax_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(wallXmax_object);

            TV ground_origin = TV(0, 0, 0) + shift;
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);
        }

        if (test_number == 777018) { // reinforced armadillo
            /*
	    ./multigrid -test 777018 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o reinforce_pn
	    ./multigrid -test 777018 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --matfree -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o reinforce_pnfree
	    ./multigrid -test 777018 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o reinforce_ours
	     */
            sim.end_frame = 48;
            sim.dx = 0.005;
            sim.gravity = -1 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            int ppc = 8;
            TV shift(2, 2, 2);

            T rho = 1000, E = 1e5, nu = 0.33;
            T rho2 = 1000, E2 = 1e8, nu2 = 0.33;

            MpmParticleHandleBase<T, dim> ph = init_helper.sampleFromVdbFile("LevelSets/armadillo_flesh.vdb", rho, ppc);
            CorotatedIsotropic<T, dim> model(E, nu);
            ph.addFBasedMpmForce(model);
            ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
            ZIRAN_INFO("p count: ", sim.particles.count);

            MpmParticleHandleBase<T, dim> ph2 = init_helper.sampleFromVdbFile("LevelSets/armadillo_bone.vdb", rho2, ppc);
            CorotatedIsotropic<T, dim> model2(E2, nu2);
            ph2.addFBasedMpmForce(model2);
            ph2.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
            ZIRAN_INFO("p count: ", sim.particles.count);

            for (int i = 0; i < 5; i++) {
                auto ball_trans = [shift, i](T time, AnalyticCollisionObject<T, dim>& object) {
                    T speed = 0.2;
                    TV velocity[5] = { TV(-speed, speed, 0), TV(speed, speed, 0), TV(speed, -speed, 0), TV(-speed, -speed, 0), TV(0, 0, 0) };
                    T t = time;
                    TV translation_velocity = velocity[i];
                    TV translation = velocity[i] * t + shift;
                    object.setTranslation(translation, translation_velocity);
                };
                TV center[5] = { TV(-0.3, 0.294, 0.141), TV(0.32, 0.266, 0.221), TV(0.2, -0.384, -0.059), TV(-0.12, -0.384, -0.075), TV(0, 0.26, -0.06) };
                T radius[5] = { .1, .1, .1, .1, .07 };
                Sphere<T, dim> ballLevelSet(center[i], radius[i]);
                AnalyticCollisionObject<T, dim> ground_object(ball_trans, ballLevelSet, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(ground_object);
            }
        }

        if (test_number == 777019) { // wheel
            /*  
	    ./multigrid -test 777019 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o wheel_ours
	    ./multigrid -test 777019 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o wheel_pn
	    ./multigrid -test 777019 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --matfree -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o wheel_pnfree
    	./multigrid -test 777019 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 --adaptiveH -o wheel_ours_adaptive

        */
            sim.end_frame = 48;
            sim.dx = 0.005;
            sim.gravity = -9.8 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt / 2;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 0;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6 / 2;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            TV shift(1.1, 1.1, 1.1);

            // Aluminum
            T nu = 0.33;
            T rho = 2.7e3;
            T E = 69e9;
            T yield = 240e6;

            // tissue
            T nu2 = 0.4;
            T rho2 = 2.7e3;
            T E2 = 1e5;

            {
                StdVector<TV> meshed_points0;
                readPositionObj("wheel.obj", meshed_points0);
                MpmParticleHandleBase<T, dim> ph = init_helper.sampleFromVdbFileWithExistingPoints(meshed_points0, "LevelSets/wheel.vdb", rho, 12);
                CorotatedIsotropic<T, dim> model(E, nu);
                model.project = HOTSettings::project;
                ph.addFBasedMpmForce(model);
                VonMisesFixedCorotated<T, dim> p(yield);
                ph.addPlasticity(model, p, "F");
                ph.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
            }

            // {
            //     T theta = M_PI / 2;
            //     CappedCylinder<T, dim> cylinder1(0.035, 0.8, Vector<T, 4>(std::cos(theta / 2), 0, 0, std::sin(theta / 2)), TV(0, 0.24, -0.02));
            //     MpmParticleHandleBase<T, dim> particles_handle2 = init_helper.sampleInAnalyticLevelSet(cylinder1, rho2, 8);
            //     CorotatedIsotropic<T, dim> model2(E2, nu2);
            //     model2.project = HOTSettings::project;
            //     particles_handle2.addFBasedMpmForce(model2);
            //     particles_handle2.transform([shift](int index, Ref<T> mass, TV& X, TV& V) { X += shift; });
            // }

            T speed = -0.2;
            auto ceilTransform = [speed](T time, AnalyticCollisionObject<T, dim>& object) {
                T t = time;
                if (t < 38 / 24.0) {
                    TV translation_velocity(0, speed, 0);
                    TV translation(0, speed * t, 0);
                    object.setTranslation(translation, translation_velocity);
                }
                else {
                    t = 38 / 24.0;
                    TV translation_velocity(0, 0, 0);
                    TV translation(0, speed * t, 0);
                    object.setTranslation(translation, translation_velocity);
                }
            };
            CappedCylinder<T, dim> cylinder1(0.08, 1, Vector<T, 4>(1, 0, 0, 0), TV(0, 1, 0) + shift);
            AnalyticCollisionObject<T, dim> ceil_object(ceilTransform, cylinder1, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ceil_object);

            TV ground_origin = TV(0, 0.01, 0) + shift;
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);

            sim.end_frame_callbacks.emplace_back([&](int frame) {
                std::string filename = sim.output_dir.absolutePath(sim.outputFileName("stress", ".bgeo"));
                Partio::ParticlesDataMutable* parts = Partio::create();
                Partio::ParticleAttribute posH, kirchhoffH, vonmisesstressH;
                posH = parts->addAttribute("position", Partio::VECTOR, 3);
                kirchhoffH = parts->addAttribute("kirchhoffStress", Partio::VECTOR, 1);
                vonmisesstressH = parts->addAttribute("vonMisesStress", Partio::VECTOR, 1);
                using TCONST = CorotatedIsotropic<T, dim>;
                using Scratch = typename TCONST::Scratch;
                AttributeName<TCONST> model_name(TCONST::name());
                AttributeName<Scratch> scratch_name(TCONST::scratch_name());
                auto& particles = sim.particles;
                StdVector<T> kirchhoffData(particles.count, T(1));
                StdVector<T> vonmisesstressData(particles.count, T(1));
                for (auto iter = particles.iter(model_name, scratch_name, F_name<T, dim>()); iter; ++iter) {
                    auto& model = iter.template get<0>();
                    auto& model_scratch = iter.template get<1>();
                    auto F = iter.template get<2>();
                    model.updateScratch(F, model_scratch);
                    TV sigma = model_scratch.sigma;
                    TV tau;
                    for (int d = 0; d < dim; d++) tau(d) = 2 * model.mu * (sigma(d) - 1) * sigma(d) + model.lambda * (model_scratch.J - 1) * model_scratch.J;
                    kirchhoffData[iter.entryId()] = tau.norm();
                    T trace_tau = tau.sum();
                    TV s = tau - TV::Ones() * (trace_tau / (T)dim);
                    vonmisesstressData[iter.entryId()] = s.norm();
                }
                for (int k = 0; k < particles.count; k++) {
                    int idx = parts->addParticle();
                    float* posP = parts->dataWrite<float>(posH, idx);
                    float* kirchhoffP = parts->dataWrite<float>(kirchhoffH, idx);
                    float* vonmisesstressP = parts->dataWrite<float>(vonmisesstressH, idx);
                    for (int d = 0; d < 3; ++d)
                        posP[d] = 0;
                    for (int d = 0; d < dim; ++d)
                        posP[d] = (float)particles.X.array[k](d);
                    kirchhoffP[0] = (float)kirchhoffData[k];
                    vonmisesstressP[0] = (float)vonmisesstressData[k];
                }
                Partio::write(filename.c_str(), *parts);
                parts->release();
            });
        }

#if 1
        if (test_number == 777020) { // turkey
            /*  
        ./multigrid -test 777020 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o turkey_ours
        ./multigrid -test 777020 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o turkey_pn
        ./multigrid -test 777020 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --matfree -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o turkey_pnfree
        ./multigrid -test 777020 --3d --usecn -cneps 1e-7 -lsolver 3 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 10000 -coarseSolver 2 -smoother 2 --l2norm -o turkey_lbfgsH
        ./multigrid -test 777020 --3d --usecn -cneps 1e-7 -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 --l2norm -o turkey_mg_pn
        */
            sim.end_frame = 185;
            sim.dx = 0.015;
            sim.gravity = -9.8 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 1;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.
            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            int ppc = 32;
            TV shift(3, 3, 3);
            // Aluminum * 1e-3
            T nu = 0.33;
            T rho = 2.7e3;
            T E = 69e9 * 3e-6;
            T yield = 240e6 * 3e-6;
            auto yogurt_transform1 = [shift](T time, AnalyticCollisionObject<T, dim>& object) {
                T t = time;
                TV translation_base = TV(0, 1.18, -0.143);
                TV translation;
                TV translation_velocity;
                // T theta = 240.0*M_PI/180.0;
                // Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2),0);
                // object.setRotation(rotation);
                // TV omega(0, theta, 0);
                // object.setAngularVelocity(omega);
                translation = translation_base + TV(-0.4 * std::sin(55 * t * M_PI / 180.0), 0., 0.5 * std::sin(120 * t * M_PI / 180.0));
                translation_velocity = TV(-0.4 * 55 * M_PI / 180.0 * std::cos(55 * t * M_PI / 180.0), 0, 0.5 * 120 * M_PI / 180.0 * std::cos(120 * t * M_PI / 180.0));
                object.setTranslation(translation, translation_velocity);
            };
            // TV source_min_corner = TV(-0.06, -0.06, -0.03) + shift;
            // TV source_max_corner = TV( 0.06,  0.06,  0.03) + shift;
            // AxisAlignedAnalyticBox<T, dim> source_sphere_ls(source_min_corner, source_max_corner);
            Sphere<T, dim> source_sphere_ls(TV(0, 0, 0) + shift, 0.09);
            TV material_speed(0.0, -0.5, 0);
            SourceCollisionObject<T, dim> box_source(yogurt_transform1, source_sphere_ls, material_speed);
            int source_id = init_helper.addSourceCollisionObject(box_source);
            init_helper.sampleSourceAtTheBeginning(source_id, rho, 16);
            sim.end_time_step_callbacks.push_back(
                [this, source_id, rho, E, nu, yield](int frame, int substep) {
                    if (frame < 144) {
                        // add more particles from source Collision object
                        int N = init_helper.sourceSampleAndPrune(source_id, rho, 16);
                        if (N) {
                            MpmParticleHandleBase<T, dim> ph = init_helper.getParticlesFromSource(source_id, rho, 16);
                            CorotatedIsotropic<T, dim> model(E, nu);
                            model.project = HOTSettings::project;
                            ph.addFBasedMpmForce(model);
                            VonMisesFixedCorotated<T, dim> p(yield);
                            ph.addPlasticity(model, p, "F");
                        }
                    }
                });
            auto turkey_transform = [shift](T time, AnalyticCollisionObject<T, dim>& object) { object.setTranslation(TV(0, 0, 0) + shift, TV(0, 0, 0)); };
            VdbLevelSet<T, dim> turkey_ls("LevelSets/turkey.vdb");
            AnalyticCollisionObject<T, dim> cherry_object(turkey_transform, turkey_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(cherry_object);
            TV ground_origin = TV(0, 0, 0) + shift;
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);
        }
#else
        if (test_number == 777020) { // turkey
            /*  
	    ./multigrid -test 777020 --3d --usecn -cneps 1e-7 --l2norm -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o turkeyreal_ours
        ./multigrid -test 777020 --3d --usecn -cneps 1e-7 -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 --l2norm -o turkeyreal_mg_pn
	    ./multigrid -test 777020 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o turkeyreal_pn
	    ./multigrid -test 777020 --3d --usecn -cneps 1e-7 --l2norm -lsolver 2 -Ainv 1 --project --linesearch --matfree -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o turkeyreal_pnfree
        ./multigrid -test 777020 --3d --usecn -cneps 1e-7 -lsolver 3 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 10000 -coarseSolver 2 -smoother 2 --l2norm -o turkeyreal_lbfgsH
        */
            sim.end_frame = 185;
            sim.dx = 0.015;
            sim.gravity = -9.8 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 1;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            //Solver params
            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            int ppc = 32;
            TV shift(3, 3, 3);

            // Aluminum * 1e-3
            T nu = 0.33;
            T rho = 2.7e3;
            T E = 69e9 * 3e-6;
            T yield = 240e6 * 3e-6;

            auto yogurt_transform1 = [shift](T time, AnalyticCollisionObject<T, dim>& object) {
                T t = time;
                TV translation_base = TV(0, 1.18, -0.143);
                TV translation;
                TV translation_velocity;

                translation = translation_base + TV(-0.4 * std::sin(55 * t * M_PI / 180.0), 0., 0.5 * std::sin(120 * t * M_PI / 180.0));
                translation_velocity = TV(-0.4 * 55 * M_PI / 180.0 * std::cos(55 * t * M_PI / 180.0), 0, 0.5 * 120 * M_PI / 180.0 * std::cos(120 * t * M_PI / 180.0));
                object.setTranslation(translation, translation_velocity);
            };

            Sphere<T, dim> source_sphere_ls(TV(0, 0, 0) + shift, 0.09);
            TV material_speed(0.0, -0.5, 0);
            SourceCollisionObject<T, dim> box_source(yogurt_transform1, source_sphere_ls, material_speed);
            int source_id = init_helper.addSourceCollisionObject(box_source);
            init_helper.sampleSourceAtTheBeginning(source_id, rho, 16);

            sim.end_time_step_callbacks.push_back(
                [this, source_id, rho, E, nu, yield](int frame, int substep) {
                    if (frame < 144) {
                        // add more particles from source Collision object
                        int N = init_helper.sourceSampleAndPrune(source_id, rho, 16);
                        if (N) {
                            MpmParticleHandleBase<T, dim> ph = init_helper.getParticlesFromSource(source_id, rho, 16);
                            CorotatedIsotropic<T, dim> model(E, nu);
                            model.project = HOTSettings::project;
                            ph.addFBasedMpmForce(model);
                            VonMisesFixedCorotated<T, dim> p(yield);
                            ph.addPlasticity(model, p, "F");
                        }
                    }
                });

            if (1) // add turkey as an elastic object
            {

                StdVector<TV> meshed_points0;
                readPositionObj("turkey.obj", meshed_points0);
                MpmParticleHandleBase<T, dim> particles_handle_top = init_helper.sampleFromVdbFileWithExistingPoints(
                    meshed_points0, "LevelSets/turkey.vdb", rho * 1.5, 8);
                auto initial_translation_kb = [&](int index, Ref<T> mass, TV& X, TV& V) {
                    X = X + shift + TV(0, 0.03, 0);
                };
                particles_handle_top.transform(initial_translation_kb);

                CorotatedIsotropic<T, dim> model2(5e7, 0.4);
                particles_handle_top.addFBasedMpmForce(model2);
            }

            else { // add turkey as a collision object
                auto turkey_transform = [shift](T time, AnalyticCollisionObject<T, dim>& object) { object.setTranslation(TV(0, 0, 0) + shift, TV(0, 0, 0)); };
                VdbLevelSet<T, dim> turkey_ls("LevelSets/turkey.vdb");
                AnalyticCollisionObject<T, dim> cherry_object(turkey_transform, turkey_ls, AnalyticCollisionObject<T, dim>::STICKY);
                init_helper.addAnalyticCollisionObject(cherry_object);
            }

            TV ground_origin = TV(0, 0, 0) + shift;
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object);
        }
#endif

        if (test_number == 888012) { // a glass of food
            /*
            ./multigrid -test 777012 --3d --usecn -cneps 0.000001 -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 1 -mg_times 1 -mg_scale 0 -coarseSolver 0 -smoother 0 -mg_omega 1.0 -dbg 0 -o pn
            ./multigrid -test 777012 --3d --usecn -cneps 0.000001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 -o ours
            ./multigrid -test 777012 --3d --usecn -cneps 0.000001 -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -mg_scale 0 -coarseSolver 2 -smoother 5 -dbg 0 --adaptiveH -o ours-ming
            */
            sim.end_frame = 240;
            sim.dx = 0.015;
            sim.gravity = -9.81 * TV::Unit(1);
            sim.step.frame_dt = (T)1 / (T)24;
            //sim.step.max_dt = sim.step.frame_dt / 24;
            //sim.step.min_dt = sim.step.max_dt;
            sim.step.max_dt = sim.step.frame_dt;
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.apic_rpic_ratio = 1;
            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.verbose = true;
            sim.cfl = 0.6;
            sim.autorestart = false;
            init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;

            int ppc = 8;

            sim.end_time_step_callbacks.push_back(
                [this, ppc](int frame, int substep) {
                    // if (frame > 1 && frame <= 120 && frame %12==0 && substep==1) {
                    if (frame < 240 && frame % 12 == 0 && substep == 1) {
                        RandomNumber<T> r(frame);
                        TM R;
                        r.randRotation(R);
                        Eigen::Quaternion<T> q(R);
                        Vector<T, 4> rotation(q.w(), q.x(), q.y(), q.z());
                        TV trans = TV(3.45, 3.1, 3.04) + TV(r.randReal(-0.1, 0.1), 0, r.randReal(-0.1, 0.1));
                        Torus<T, dim> torus(.08, .08 / 3, rotation, trans);
                        // Sphere<T, dim> sphere(TV(3.45, 3.1, 3.04), .05);

                        if (1) {
                            T rho = 1000, E = 1e5;
                            MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(torus, rho, ppc);
                            CorotatedIsotropic<T, dim> model(E, .35);
                            //SnowPlasticity<T> p(5, 0.01, 0.001, 0.0001, 1000);
                            SnowPlasticity<T> p(0, 0.01, 0.001, -2, 5);
                            ph.addPlasticity(model, p, "F");
                            model.project = HOTSettings::project;
                            ph.addFBasedMpmForce(model);
                        }
                        else {
                            T rho = 1000, E = 2.5e4;
                            MpmParticleHandleBase<T, dim> ph = init_helper.sampleInAnalyticLevelSet(torus, rho, ppc);
                            CorotatedIsotropic<T, dim> model(10000, .4);
                            model.mu = 0;
                            model.project = HOTSettings::project;
                            ph.addFBasedMpmForce(model);
                        }
                    }
                });

            CappedCylinder<T, dim> cylinder1(0.33, 1, Vector<T, 4>(1, 0, 0, 0), TV(3.45, 2.71, 3.04));
            CappedCylinder<T, dim> cylinder2(0.27, 1, Vector<T, 4>(1, 0, 0, 0), TV(3.45, 3.07, 3.04));
            DifferenceLevelSet<T, dim> materialRegionLevelSet;
            materialRegionLevelSet.add(cylinder1, cylinder2);
            AnalyticCollisionObject<T, dim> bowl(materialRegionLevelSet, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(bowl);
        }

        if (test_number == 888001) {
            //sim.output_dir.path = "output/FCR_cube_stack";
            sim.end_frame = 120;
            sim.dx = 1e-2;
            sim.gravity = -9.8 * TV::Unit(1);
            sim.symplectic = false;
            sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
            sim.verbose = true;
            sim.cfl = 0.5;
            sim.autorestart = false;

            sim.newton.tolerance = 1e-4;
            sim.newton.max_iterations = 10000;
            sim.lbfgs.tolerance = 1e-4;
            sim.lbfgs.max_iterations = 10000;
            //sim.quasistatic = false;
            sim.objective.minres.max_iterations = 10000;
            sim.objective.minres.tolerance = 1e-4;
            sim.objective.cg.max_iterations = 10000;
            sim.objective.cg.tolerance = 1e-4;
            // sim.newton.linear_solve_tolerance_scale = 1e7; ///< use absolute tolerance

            sim.objective.matrix_free = HOTSettings::matrixFree;
            sim.write_substeps = false;
            sim.step.frame_dt = (T)1 / (T)24;

            sim.step.max_dt = sim.step.frame_dt;

            T gelatinDensity = 1300;
            T beefDensity = 950;
            T rho_soft_rubber = 2e3;
            T youngs_soft_rubber = 5e5;
            T rho_wood = 2e3;
            T youngs_wood = 1e8;
            T E[2] = { youngs_wood, youngs_soft_rubber };
            T nu = 0.4;

            T Elen = 1.0;
            TV origin(10.0, Elen, 10.0);
            int ppc = 8;

            for (int i = 0; i < 3; ++i) {
                T theta = 0;
                Vector<T, 4> rotation(std::cos(theta / 2), std::sin(theta / 2), 0, 0);
                TV translation(0, 0, 0);
                CappedCylinder<T, dim> cylinder(/*radius*/ Elen * 0.5, /*height*/ Elen * 0.2, rotation, origin);

                MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(cylinder, i == 1 ? rho_wood : rho_soft_rubber, ppc);

                //LinearCorotated<T, dim> model(E[i % 2], nu[i % 2]);
                CorotatedIsotropic<T, dim> model(E[(i + 1) % 2], nu);
                model.project = HOTSettings::project;
                //particles_handle.addLinearCorotatedMpmForce(model);
                particles_handle.addFBasedMpmForce(model);

                origin[1] += Elen * 0.3;
            }

            TV ground_origin(0, sim.dx * 10, 0);
            TV ground_normal(0, 1, 0);
            HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
            //AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SLIP);
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
            init_helper.addAnalyticCollisionObject(ground_object); // add ground

            sim.output_dir.path = destination("output/FCR_ham", sim.step.max_dt, sim.dx, E[1], ppc, sim.particles.count);
        }

        // #############################################################################################
        if (CaseSettings::output_folder.length())
            sim.output_dir.path = "output/" + CaseSettings::output_folder;
        // #############################################################################################
    }
};
} // namespace ZIRAN
#endif
