#include "MultigridSimulation.h"

namespace ZIRAN {

    template <class T, int dim>
    void MultigridSimulation<T, dim>::mark_colors(const std::vector<Vector<int, dim>>& id2coord, const std::unique_ptr<Coord2IdMap>& coord2id, std::vector<std::array<int, 3>>& colorOrder, std::array<std::vector<std::vector<int>>, 8>& coloredBlockDofs)
    {
        std::array<int, 8> blockCnts;
        for (auto& bc : blockCnts)
            bc = 0;
        std::array<std::unordered_map<ULL, int>, 8> blockIds;
        //std::map<IV, IndexType, CoordComp<IndexType, dim>> blockIds;
        for (int i = 0; i < (int)id2coord.size(); ++i) {
            int blockIndices[dim];
            for (int d = 0; d < dim; ++d)
                blockIndices[d] = id2coord[i][d] >> 2;
            int color = ((blockIndices[0] & 1) << 2) | ((blockIndices[1] & 1) << 1) | (blockIndices[2] & 1);
            ULL hash_key = (ULL)blockIndices[0] * hash_seed * hash_seed + (ULL)blockIndices[1] * hash_seed + (ULL)blockIndices[2];
            //IV hash_key{ blockIndices[0], blockIndices[1], blockIndices[2] };
            int blockId;
            if (blockIds[color].find(hash_key) == blockIds[color].end()) {
                blockIds[color][hash_key] = blockCnts[color]++;
                coloredBlockDofs[color].emplace_back();
            }
            blockId = blockIds[color][hash_key];
            auto& blockNodes = coloredBlockDofs[color];
            Coord2IdMap::const_accessor rec;
            ZIRAN_ASSERT(coord2id->find(rec, (ULL)id2coord[i][0] * hash_seed * hash_seed + (ULL)id2coord[i][1] * hash_seed + (ULL)id2coord[i][2]));
            blockNodes[blockId].push_back(rec->second);
            /// for symmetric Gauss-Seidel
            colorOrder[rec->second][0] = color;
            colorOrder[rec->second][1] = blockId;
            colorOrder[rec->second][2] = blockNodes[blockId].size();
        }
    }

    template <class T, int dim>
    void MultigridSimulation<T, dim>::sortParticlesAndPolluteMultigrid(BaselineMultigrid& multigrid)
    {
        ZIRAN_TIMER();

        auto& particle_sorter{ multigrid.particle_sorter };
        auto& particle_base_offset{ multigrid.particle_base_offset };
        auto& particle_order{ multigrid.particle_order };

        //ZIRAN_INFO("begin sorting particles with ", particle_sorter.size(), " entries");
        auto& particles = Base::particles;
        using SparseMask = typename MpmGrid<T, dim>::SparseMask;
        auto& Xarray = particles.X.array;

        constexpr int index_bits = (32 - SparseMask::block_bits);
        ZIRAN_ASSERT(particles.count < (1 << index_bits));

        ZIRAN_INFO(particle_sorter.size(), " -> ", particles.count);
        if (particles.count != (int)particle_sorter.size()) {
            particle_base_offset.resize(particles.count);
            particle_sorter.resize(particles.count);
            particle_order.resize(particles.count);
        }

        auto dx = multigrid.dx;
        T one_over_dx = (T)1 / dx;
        tbb::parallel_for(0, particles.count, [&](int i) {
            uint64_t offset = SparseMask::Linear_Offset(to_std_array(
                baseNode<Base::interpolation_degree, T, dim>(Xarray[i] * one_over_dx)));
            particle_sorter[i] = ((offset >> SparseMask::data_bits) << index_bits) + i;
        });

        tbb::parallel_sort(particle_sorter.begin(), particle_sorter.end());

        auto& particle_group{ multigrid.particle_group };
        auto& block_offset{ multigrid.block_offset };
        particle_group.clear();
        block_offset.clear();
        int last_index = 0;
        for (int i = 0; i < particles.count; ++i)
            if (i == particles.count - 1 || (particle_sorter[i] >> 32) != (particle_sorter[i + 1] >> 32)) {
                particle_group.push_back(std::make_pair(last_index, i));
                block_offset.push_back(particle_sorter[i] >> 32);
                last_index = i + 1;
            }

        auto& grid = multigrid.grid;
        grid.page_map->Clear();
        for (int i = 0; i < particles.count; ++i) {
            particle_order[i] = (int)(particle_sorter[i] & ((1ll << index_bits) - 1));
            uint64_t offset = (particle_sorter[i] >> index_bits) << SparseMask::data_bits;
            particle_base_offset[particle_order[i]] = offset;
            if (i == particles.count - 1 || (particle_sorter[i] >> 32) != (particle_sorter[i + 1] >> 32)) {
                grid.page_map->Set_Page(offset);
                if constexpr (dim == 2) {
                    auto x = 1 << SparseMask::block_xbits;
                    auto y = 1 << SparseMask::block_ybits;
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j)
                            grid.page_map->Set_Page(SparseMask::Packed_Add(
                                offset, SparseMask::Linear_Offset(x * i, y * j)));
                }
                else {
                    auto x = 1 << SparseMask::block_xbits;
                    auto y = 1 << SparseMask::block_ybits;
                    auto z = 1 << SparseMask::block_zbits;
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j)
                            for (int k = 0; k < 2; ++k)
                                grid.page_map->Set_Page(SparseMask::Packed_Add(
                                    offset, SparseMask::Linear_Offset(x * i, y * j, z * k)));
                }
            }
        }
        grid.page_map->Update_Block_Offsets();

        auto grid_array = grid.grid->Get_Array();
        auto blocks = grid.page_map->Get_Blocks();
        for (int b = 0; b < (int)blocks.second; ++b) {
            auto base_offset = blocks.first[b];
            std::memset(&grid_array(base_offset), 0, (size_t)(1 << MpmGrid<T, dim>::log2_page));
            GridState<T, dim>* g = reinterpret_cast<GridState<T, dim>*>(&grid_array(base_offset));
            for (int i = 0; i < (int)SparseMask::elements_per_block; ++i)
                g[i].idx = -1;
        }
    }

    template <class T, int dim>
    void MultigridSimulation<T, dim>::buildMultigridBoundaries(BaselineMultigrid& multigrid)
    {
        ZIRAN_TIMER();
        auto& grid{ multigrid.grid };
        auto& collision_nodes{ multigrid.collision_nodes };
        auto& collision_objects{ Base::collision_objects };
        grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            TV vi = TV::Zero();
            TV wn;

            const TV xi = node.template cast<T>() * multigrid.dx;
            TM normal_basis, R;

            bool any_collision = AnalyticCollisionObject<T, dim>::multiObjectCollision(collision_objects, xi, vi, normal_basis, wn);

            int node_id = g.idx;
            assert(node_id < multigrid.num_nodes);
            if (any_collision) {
#if 1
                bool isSlip = wn != TV::Zero();
                if (isSlip) {
                    R = RotationExtractor<T, dim>::rotate(wn);
                    //ZIRAN_INFO(isSlip ? "SLIP!\n" : "STICKY!\n");
                    //printf("normal: %e\t%e\t%e\n\n%e\t%e\t%e\n%e\t%e\t%e\n%e\t%e\t%e\n\n", wn(0), wn(1), wn(2), R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1), R(2, 2));
                    //getchar();
                }
                else
                    R = TM::Identity();
                CollisionNode<T, dim> Z{ node_id, TM::Identity() - normal_basis * normal_basis.transpose(), /* wn, */ R, R.inverse(), isSlip };
#else
                CollisionNode<T, dim> Z{ node_id, TM::Identity() - normal_basis * normal_basis.transpose() };
#endif
                collision_nodes.push_back(Z);
            }
        });
    }

    template <class T, int dim>
    void MultigridSimulation<T, dim>::buildMultigridMatrices(BaselineMultigrid& multigrid, std::unique_ptr<SpMat>& resmat, std::unique_ptr<SpMat>& sysmat, std::unique_ptr<SpMat>& promat, std::unique_ptr<Coord2IdMap>& finerCoord2Id, std::vector<IV>& validFinerNodes)
    {
        ZIRAN_TIMER();
        auto& particles = Base::particles;

#define kernel_range 2
        auto wts = linear_weight_template<T, dim>();

        std::vector<IV> new_id2coord;
        new_id2coord.resize(multigrid.num_nodes);
        std::vector<std::pair<ULL, int>> coord_id_seq;

        coord_id_seq.resize(multigrid.num_nodes);
        multigrid.grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            ULL hash_key = (ULL)node[0] * hash_seed * hash_seed + (ULL)node[1] * hash_seed + (ULL)node[2];
            coord_id_seq[g.idx] = std::make_pair(hash_key, g.idx);
            new_id2coord[g.idx] = node;
        });
        auto new_coord2id = std::make_unique<Coord2IdMap>(coord_id_seq.begin(), coord_id_seq.end());

        promat->colsize = pow(kernel_range, dim);
        promat->entryCol.resize(finerCoord2Id->size() * pow(kernel_range, dim));
        promat->entryVal.resize(finerCoord2Id->size() * pow(kernel_range, dim));
        /// initialize zero mat3x3
        std::fill(promat->entryVal.begin(), promat->entryVal.end(), TM::Zero());
        std::fill(promat->entryCol.begin(), promat->entryCol.end(), 0);
        for (int i = 0; i < (int)validFinerNodes.size(); ++i) {
            Coord2IdMap::const_accessor rec;
            const IV& coord = validFinerNodes[i];
            int x = coord(0), y = coord(1), z = coord(2);
            finerCoord2Id->find(rec, (ULL)x * hash_seed * hash_seed + (ULL)y * hash_seed + (ULL)z);
            int finer_id = rec->second;

            const auto& wt = wts[(x & 1) * 4 + (y & 1) * 2 + (z & 1)];
            for (int new_x = x / 2; new_x <= x / 2 + 1; ++new_x)
                for (int new_y = y / 2; new_y <= y / 2 + 1; ++new_y)
                    for (int new_z = z / 2; new_z <= z / 2 + 1; ++new_z) {
                        int idx = (new_x - x / 2 + 1) * 9 + (new_y - y / 2 + 1) * 3 + new_z - z / 2 + 1;
                        int linear_idx = (new_x - x / 2) * 4 + (new_y - y / 2) * 2 + new_z - z / 2;
                        T weight = wt[idx];
                        if (weight == 0) {
                            promat->entryCol[finer_id * pow(kernel_range, dim) + linear_idx] = promat->entryCol[finer_id * pow(kernel_range, dim)];
                            promat->entryVal[finer_id * pow(kernel_range, dim) + linear_idx] = TM::Zero();
                            continue;
                        }
                        //new_id2coord.push_back(IV{ new_x, new_y, new_z });
                        ULL hash_key = (ULL)new_x * hash_seed * hash_seed + (ULL)new_y * hash_seed + (ULL)new_z;
                        //IV hash_key{ new_x, new_y, new_z };
                        ZIRAN_ASSERT(new_coord2id->find(rec, hash_key));
                        int coarser_id = rec->second;
                        //ZIRAN_ASSERT(linear_idx >= 0 && linear_idx < 8, "linear offset computation wrong!");
                        promat->entryCol[finer_id * pow(kernel_range, dim) + linear_idx] = coarser_id;
                        promat->entryVal[finer_id * pow(kernel_range, dim) + linear_idx] = weight * TM::Identity();
                    }
        }
        resmat->buildTransposeMatrix(*promat, multigrid.num_nodes);

        /// system matrix
        sysmat->colsize = pow(5, dim);
        auto& entryCol = sysmat->entryCol;
        auto& entryVal = sysmat->entryVal;
        entryCol.resize(multigrid.num_nodes * 125);
        entryVal.resize(multigrid.num_nodes * 125);
        sysmat->allocateColorIds();
        //ZIRAN_INFO("Prepare color partition!");
        mark_colors(new_id2coord, new_coord2id, sysmat->colorOrder, sysmat->coloredBlockDofs);
        //ZIRAN_INFO("Finish system matrix SGS precalc!");
        validFinerNodes.swap(new_id2coord);
        finerCoord2Id = std::move(new_coord2id);
        //ZIRAN_INFO("Finish prolongation & restriction matrix build!");

        // inertia term
        T inertia_scale = T(1);
        multigrid.grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            for (int i = 0; i < 125; ++i) {
                entryCol[g.idx * 125 + i] = 0;
                entryVal[g.idx * 125 + i] = TM::Zero();
            }
            entryCol[g.idx * 125 + objective.linearOffset(IV::Zero())] = g.idx;
            entryVal[g.idx * 125 + objective.linearOffset(IV::Zero())] = inertia_scale * multigrid.mass_matrix(g.idx) * TM::Identity();
        });

        // force term
        T force_scale = Base::dt * Base::dt;
        auto* vol_pointer = &particles.DataManager::get(AttributeName<T>("element measure"));

        for (auto& h : multigrid.force_helpers)
            h->runLambdaWithDifferential(multigrid.particle_order, multigrid.particle_group, multigrid.block_offset,
                [&](int i, const Hessian& ddF, const TM& Fn, const T& ddJ, const T& Jn, bool use_J) {
                    auto& Xp = particles.X[i];
                    auto& vol = (*vol_pointer)[i];

                    BSplineWeights<T, dim> spline(Xp, multigrid.dx);
                    std::vector<TV> cached_w(27);
                    std::vector<IV> cached_node(27);
                    std::vector<int> cached_idx(27);
                    int cnt = 0;
                    multigrid.grid.iterateKernel(spline, multigrid.particle_base_offset[i], [&](const IV& node, const T& w, const TV& dw, GridState<T, dim>& g) {
                        if (g.idx < 0)
                            return;
                        cached_w[cnt] = use_J ? dw : Fn.transpose() * dw;
                        cached_node[cnt] = node;
                        cached_idx[cnt++] = g.idx;
                    });

                    for (int i = 0; i < cnt; ++i) {
                        const TV& wi = cached_w[i];
                        const IV& nodei = cached_node[i];
                        const int& dofi = cached_idx[i];
                        ///
                        for (int j = 0; j < cnt; ++j) {
                            const TV& wj = cached_w[j];
                            const IV& nodej = cached_node[j];
                            const int& dofj = cached_idx[j];
                            if (dofj < dofi) continue;
                            TM dFdX = TM::Zero();
                            if (use_J)
                                dFdX += ddJ * Jn * Jn * wi * wj.transpose();
                            else
                                for (int q = 0; q < dim; q++)
                                    for (int v = 0; v < dim; v++) {
                                        dFdX += ddF.template block<dim, dim>(dim * v, dim * q) * wi(v) * wj(q);
                                    }
                            TM delta = force_scale * vol * dFdX;
                            ///

                            entryCol[dofi * 125 + objective.linearOffset(nodei - nodej)] = dofj;
                            entryVal[dofi * 125 + objective.linearOffset(nodei - nodej)] += delta;
                            if (dofi != dofj) {
                                entryCol[dofj * 125 + objective.linearOffset(nodej - nodei)] = dofi;
                                entryVal[dofj * 125 + objective.linearOffset(nodej - nodei)] += delta.transpose();
                            }
                        }
                    }
                },
                HOTSettings::useBaselineMultigrid ? 2 : 0);

        if constexpr (true) { ///< must do boundary condition projection
            std::map<int, const CollisionNode<T, dim>&> bCollide;
            for (auto iter = multigrid.collision_nodes.begin(); iter != multigrid.collision_nodes.end(); ++iter) {
                bCollide.emplace(iter->node_id, *iter);
                //bCollide[iter->node_id] = *iter;
            }
            tbb::parallel_for(0, multigrid.num_nodes, [&](int i) {
                int st = i * 125;
                int ed = st + 125;
                const auto& iIter = bCollide.find(i);
                bool iCollide = iIter != bCollide.end();
                bool iSlip = iCollide ? iIter->second.shouldRotate : false;
                for (; st < ed; ++st) {
                    int j = entryCol[st];
                    if (j == -1) { ///< zero-entry
                        entryCol[st] = i > 0 ? 0 : 1;
                        continue;
                    }
                    const auto& jIter = bCollide.find(j);
                    bool jCollide = jIter != bCollide.end();
                    if (!iCollide && !jCollide) ///< no projection
                        continue;
                    bool jSlip = jCollide ? jIter->second.shouldRotate : false;
                    auto& val = entryVal[st];
                    if ((iCollide && !iSlip) || (jCollide && !jSlip)) { ///< sticky projection
                        //dRhs.col(j) -= val * rhs.col(j);
                        if (j == i)
                            val = TM::Identity();
                        else
                            val = TM::Zero();
                        continue;
                    }
                    if (iSlip) val = iIter->second.R * val;
                    if (jSlip) val = val * jIter->second.Rinv;
                    if (iSlip) val(0, 0) = 0, val(0, 1) = 0, val(0, 2) = 0;
                    if (jSlip) val(0, 0) = 0, val(1, 0) = 0, val(2, 0) = 0;
                    if (i == j) val(0, 0) = 1;
                }
            });
        }
        sysmat->buildDiagonal(HOTSettings::Ainv);
        ZIRAN_INFO("Allocate color partition!");
    }

    template <class T, int dim>
    void MultigridSimulation<T, dim>::particlesToMultigrids(MGOperator& mgp)
    {
        ZIRAN_TIMER();
        using TM4 = Matrix<T, 4, 4>;
        using TV4 = Vector<T, 4>;
        T curdx = Base::dx;
        while ((int)multigrids.size() + 1 < HOTSettings::levelCnt) {
            multigrids.emplace_back(Base::force->helpers);
            multigrids.back().dx = (curdx *= 2);
        }
        printf("\tmultigrid size %d\n", (int)multigrids.size());
        auto& Xarray = Base::particles.X.array;
        auto& Varray = Base::particles.V.array;
        auto& marray = Base::particles.mass.array;
        const StdVector<Matrix<T, dim, dim>>* Carray_pointer;
        Carray_pointer = ((Base::transfer_scheme == Base::APIC_blend_RPIC) && Base::interpolation_degree != 1) ? (&(Base::particles.DataManager::get(C_name<TM>()).array)) : NULL;

        /// finest level
        std::vector<int> dofs(HOTSettings::levelCnt);
        std::vector<std::unique_ptr<SpMat>> resmats(HOTSettings::levelCnt - 1);
        std::vector<std::unique_ptr<SpMat>> sysmats(HOTSettings::levelCnt);
        std::vector<std::unique_ptr<SpMat>> promats(HOTSettings::levelCnt - 1);
        dofs[0] = Base::grid.getNumNodes();
        //sysmats[0]->allocateColorIds();
        //markColors(sysmats[0]->colorOrder, sysmats[0]->coloredBlockDofs);

        /// mapping
        std::vector<IV> id2coord;
        std::unique_ptr<Coord2IdMap> coord2id;
        std::vector<std::pair<ULL, int>> coord_id_seq;
        id2coord.resize(Base::num_nodes);
        coord_id_seq.resize(Base::num_nodes);
        Base::grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            id2coord[g.idx] = node;
            ULL hash_key = (ULL)node[0] * hash_seed * hash_seed + (ULL)node[1] * hash_seed + (ULL)node[2];
            coord_id_seq[g.idx] = std::make_pair(hash_key, g.idx);
        });
        coord2id = std::make_unique<Coord2IdMap>(coord_id_seq.begin(), coord_id_seq.end());

        sysmats[0] = std::make_unique<SpMat>();
        sysmats[0]->colsize = pow(5, dim);
        sysmats[0]->entryCol = objective.entryCol;
        sysmats[0]->entryVal = objective.entryVal;
        sysmats[0]->buildDiagonal(HOTSettings::Ainv);
        sysmats[0]->allocateColorIds();
        mark_colors(id2coord, coord2id, sysmats[0]->colorOrder, sysmats[0]->coloredBlockDofs);

        for (int i = 0; i < (int)multigrids.size(); ++i) {
            auto& multigrid{ multigrids[i] };

            auto& grid{ multigrid.grid };
            auto grid_array = grid.grid->Get_Array();
            auto& particle_group{ multigrid.particle_group };
            auto& block_offset{ multigrid.block_offset };
            auto& particle_order{ multigrid.particle_order };
            auto& particle_base_offset{ multigrid.particle_base_offset };
            auto& mass_matrix{ multigrid.mass_matrix };

            ZIRAN_INFO("Building multigrid ", i + 1);
            //multigrid.clear();
            if (objective.isNewStep == true) {
                sortParticlesAndPolluteMultigrid(multigrid);
                /// grid mass
                {
                    ZIRAN_TIMER();
                    ZIRAN_INFO("particle group (for multigrid) size ", particle_group.size());
                    for (uint64_t color = 0; color < (1 << dim); ++color) {
                        tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
                            if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
                                return;
                            for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                                int i = particle_order[idx];
                                TV& Xp = Xarray[i];
                                T mass = marray[i];
                                BSplineWeights<T, dim> spline(Xp, multigrid.dx);
                                grid.iterateKernel(spline, particle_base_offset[i], [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                                    g.m += w * mass;
                                });
                            }
                        });
                    }
                    ZIRAN_WARN("Just finish multigrid mass p2g");
                }

                multigrid.num_nodes = grid.getNumNodes();
                mass_matrix.resize(multigrid.num_nodes, 1);
                grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
                    mass_matrix(g.idx) = g.m;
                });

                buildMultigridBoundaries(multigrid);
            }
            dofs[i + 1] = multigrid.num_nodes;
            printf("higher level node count: %d\n", multigrid.num_nodes);
            promats[i] = std::make_unique<SpMat>();
            sysmats[i + 1] = std::make_unique<SpMat>();
            resmats[i] = std::make_unique<SpMat>();
            buildMultigridMatrices(multigrid, resmats[i], sysmats[i + 1], promats[i], coord2id, id2coord);
        }

        MGOperator::scaler_func = MGOperator::scale_diagonal_block_inverse;
        mgp.regular.smoothFunc = MGOperator::gs_smooth;
        mgp.top.smoothFunc = MGOperator::cg_smooth;
        mgp.top.tolFunc = [](int level) { return HOTSettings::cneps * HOTSettings::cneps; };
        mgp.regular.tolFunc = [](int level) { return 0; }; /// small enough to ensure smooth times
        mgp.downIterFunc = [](int level) { return HOTSettings::times + level * HOTSettings::levelscale; };
        mgp.splitLevel = HOTSettings::levelCnt - 1;
        mgp.upIterFunc = [](int level) { return HOTSettings::times + level * HOTSettings::levelscale; };
        mgp.topIterFunc = [](int level) { return 10000; };
        mgp.init(dofs, resmats, sysmats, promats);
        mgp.correctResidualProjection = [](TVStack&) {};
    }

}