#ifndef SQUARE_MATRIX_H
#define SQUARE_MATRIX_H

#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/ArpackSupport>
#include <time.h>

namespace ZIRAN {

template <class T, int dim>
class SquareMatrix {
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using IV = std::array<int, 3>;
    using Vec = Vector<T, Eigen::Dynamic>;
    using ColSpMat = Eigen::SparseMatrix<T, Eigen::ColMajor>;
    using ColSpIter = typename ColSpMat::InnerIterator;

public:
    SquareMatrix()
    {
        colsize = 0;
    }

    int colsize;
    std::vector<int> entryCol;
    std::vector<TM> entryVal;
    std::vector<TM> diagonalVal;
    std::vector<TM> diagonalEntry;
    std::vector<TM> diagonalBlock;
    std::array<std::vector<std::vector<int>>, (1 << dim)> coloredBlockDofs;
    std::vector<IV> colorOrder;
    Eigen::IncompleteCholesky<T> ICSolver;
    ColSpMat eigenMat;
    T lMin = 1e-8, lMax = 1e2;

    static int comp(const IV& a, const IV& b)
    {
        for (int i = 0; i < 3; ++i)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
    }

    auto rows() const
    {
        return entryVal.size() / colsize;
    }

    void allocateColorIds()
    {
        colorOrder.resize(rows());
        for (auto& blockDofs : coloredBlockDofs)
            blockDofs.clear();
    }

    int findEntry(int i, int j)
    {
        int st = i * colsize;
        int ed = st + colsize;
        for (; st < ed; ++st) {
            if (entryVal[st].norm() < 1e-60) continue;
            if (entryCol[st] == j) return st;
        }
        return -1;
    }

    void coloringSanityCheck(std::vector<Vector<int, dim>>& id2coord)
    {
        int rowcnt = rows();
        for (int i = 0; i < rowcnt; ++i) {
            int st = i * colsize;
            int ed = st + colsize;
            for (; st < ed; ++st) {
                int j = entryCol[st];
                TM val = entryVal[st];
            }
        }
    }

    void symmetricSanityCheck()
    {
        int rowcnt = rows();
        for (int i = 0; i < rowcnt; ++i) {
            int st = i * colsize;
            int ed = st + colsize;
            for (; st < ed; ++st) {
                int j = entryCol[st];
                TM val = entryVal[st];
                if (val == TM::Zero()) continue;
                int symentry = findEntry(j, i);
                ZIRAN_ASSERT(symentry != -1, "no symmetric entry (", i, "(", st, "), ", j, "), ", "for the system matrix!\n",
                    val(0, 0), "\t\t", val(0, 1), "\t\t", val(0, 2), "\n",
                    val(1, 0), "\t\t", val(1, 1), "\t\t", val(1, 2), "\n",
                    val(2, 0), "\t\t", val(2, 1), "\t\t", val(2, 2));
                TM symval = entryVal[symentry];
                ZIRAN_ASSERT((val - symval.transpose()).norm() < 1e-10, "symmetric entry (", i, "(", st, "), ", j, "(", symentry, ")) value mismatch!\n",
                    val(0, 0), "\t\t", val(0, 1), "\t\t", val(0, 2), "\n",
                    val(1, 0), "\t\t", val(1, 1), "\t\t", val(1, 2), "\n",
                    val(2, 0), "\t\t", val(2, 1), "\t\t", val(2, 2), " vs.\n",
                    symval(0, 0), "\t\t", symval(0, 1), "\t\t", symval(0, 2), "\n",
                    symval(1, 0), "\t\t", symval(1, 1), "\t\t", symval(1, 2), "\n",
                    symval(2, 0), "\t\t", symval(2, 1), "\t\t", symval(2, 2));
            }
        }
    }

    T innerProduct(const TVStack& ddv, const TVStack& residual) const
    {
        ZIRAN_ASSERT(residual.cols() == ddv.cols());
        T result = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, residual.cols(), 256), (T)0,
            [&](const tbb::blocked_range<int>& range, T ns) -> T {
                int start = range.begin();
                int length = (range.end() - range.begin());
                const auto& ddv_block = ddv.middleCols(start, length);
                const auto& r_block = residual.middleCols(start, length);

                T to_add = ((ddv_block.transpose() * r_block).diagonal().array()).sum();
                return ns += to_add;
            },
            [](T a, T b) -> T { return a + b; });
        return result;
    }

    void PDSanityCheck()
    {
        int n = rows();
        T maxentry = 1e2;
        TVStack r, t;
        r.resize(dim, n);
        t.resize(dim, n);
        srand((unsigned)time(0));
        for (int i = 0; i < n / 200; ++i) {
            /// gen random vector
            for (int j = 0; j < n; j++)
                for (int d = 0; d < dim; ++d) {
                    T rnd = (T)rand() / (RAND_MAX);
                    r(d, j) = (rnd - 0.5) * maxentry;
                }
            multiply(r, t);
            T ag = innerProduct(r, t) / r.norm() / t.norm();
            if (ag <= 0) {
                ZIRAN_WARN("Testing PD: occurs negative xAx^T ", ag, "!");
                getchar();
                return;
            }
        };
        ZIRAN_INFO("finishes random PD test for system matrix!");
    }

    void SPDSanityCheck()
    {
        using ColSpMat = Eigen::SparseMatrix<T, Eigen::ColMajor>;
        using ColSpIter = typename ColSpMat::InnerIterator;
        std::vector<Eigen::Triplet<T>> triplets;
        int rowcnt = rows();
        ZIRAN_INFO("current matrix has ", rowcnt, " dofs!");
        if (rowcnt > 5000) { ///< use cheap random PD check
            PDSanityCheck();
            ZIRAN_INFO("finish PD sanity check");
            return;
        }
        double minentry = 1e10;
        for (int i = 0; i < rowcnt; ++i) {
            int st = i * colsize;
            int ed = st + colsize;
            for (; st < ed; ++st) {
                int j = entryCol[st];
                TM val = entryVal[st];
                for (int a = 0; a < dim; ++a)
                    for (int b = 0; b < dim; ++b) {
                        triplets.emplace_back(i * dim + a, j * dim + b, val(a, b));
                        auto e = std::abs(val(a, b));
                        if (e < minentry && e > 0)
                            minentry = e;
                    }
            }
        }
        ZIRAN_INFO("minimum entry value is ", minentry);
        ColSpMat mat;
        mat.resize(rowcnt * dim, rowcnt * dim);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        typedef Eigen::SimplicialLDLT<ColSpMat> SparseChol;
        typedef Eigen::ArpackGeneralizedSelfAdjointEigenSolver<ColSpMat, SparseChol> Arpack;
        Arpack arpack;
        int nbrEigenvalues = 2;
        arpack.compute(mat, nbrEigenvalues, "SA");
        std::cout << "arpack eigenvalues of system matrix\n"
                  << arpack.eigenvalues().transpose() << std::endl;
    }

    void outputMatrix()
    {
        using ColSpMat = Eigen::SparseMatrix<T, Eigen::ColMajor>;
        using ColSpIter = typename ColSpMat::InnerIterator;

        int rowcnt = rows();
        ZIRAN_INFO("current matrix has ", rowcnt, " dofs!");

        FILE* out = fopen(("A" + std::to_string(rowcnt)).c_str(), "w");
        ZIRAN_ASSERT(out);
        for (int i = 0; i < rowcnt; ++i) {
            int st = i * colsize;
            int ed = st + colsize;
            for (; st < ed; ++st) {
                int j = entryCol[st];
                const TM& val = entryVal[st];
                for (int a = 0; a < dim; ++a)
                    for (int b = 0; b < dim; ++b) {
                        if (val(a, b))
                            fprintf(out, "%d %d %le\n", i * dim + a + 1, j * dim + b + 1, val(a, b));
                    }
            }
        }
        fclose(out);

        ZIRAN_INFO("output done");
    }

    void setupICSolver()
    {
        std::vector<Eigen::Triplet<T>> triplets;
        int rowcnt = rows();
        ZIRAN_INFO("START BUILDING EIGEN SPARSE MATRIX");
        for (int i = 0; i < rowcnt; ++i) {
            int st = i * colsize;
            int ed = st + colsize;
            for (; st < ed; ++st) {
                int j = entryCol[st];
                const TM& val = entryVal[st];
                for (int a = 0; a < dim; ++a)
                    for (int b = 0; b < dim; ++b) {
                        if (val(a, b) != 0.0)
                            triplets.emplace_back(i * dim + a, j * dim + b, val(a, b));
                    }
            }
        }
        eigenMat.resize(rowcnt * dim, rowcnt * dim);
        eigenMat.setFromTriplets(triplets.begin(), triplets.end());
        ZIRAN_INFO("END BUILDING EIGEN SPARSE MATRIX");
        ICSolver.analyzePattern(eigenMat);
        ZIRAN_INFO("END ANALYZING EIGEN SPARSE MATRIX");
        ICSolver.factorize(eigenMat);
        ZIRAN_INFO("END FACTORIZING EIGEN SPARSE MATRIX");
        //ICSolver.compute(mat);
    }

    void solveIC(const TVStack& rhs, TVStack& sol)
    {
        using ZIRAN::EIGEN_EXT::vec;
        vec(sol) = ICSolver.solve(vec(rhs));
    }

    void solve(const TVStack& rhs, TVStack& sol)
    {
        using ColSpMat = Eigen::SparseMatrix<T, Eigen::ColMajor>;
        using ColSpIter = typename ColSpMat::InnerIterator;
        std::vector<Eigen::Triplet<T>> triplets;
        int rowcnt = rows();
        double minentry = 1e10;
        ZIRAN_INFO("START BUILDING EIGEN SPARSE MATRIX");
        for (int i = 0; i < rowcnt; ++i) {
            int st = i * colsize;
            int ed = st + colsize;
            for (; st < ed; ++st) {
                int j = entryCol[st];
                const TM& val = entryVal[st];
                for (int a = 0; a < dim; ++a)
                    for (int b = 0; b < dim; ++b) {
                        if (val(a, b) != 0.0)
                            triplets.emplace_back(i * dim + a, j * dim + b, val(a, b));
                    }
            }
        }
        ColSpMat mat;
        mat.resize(rowcnt * dim, rowcnt * dim);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        ZIRAN_INFO("END BUILDING EIGEN SPARSE MATRIX");

        using ZIRAN::EIGEN_EXT::vec;
        // Eigen::SimplicialLDLT<Eigen::SparseMatrix<T> > m_solver;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<T>> m_solver;
        Timer t;
        T time;
        t.start();
        ZIRAN_INFO("BEGIN DIRECT SOLVING!");
        m_solver.compute(mat);
        time += t.click(false).count();
        ZIRAN_INFO("System matrix factorization takes ", time, "s");
        ZIRAN_ASSERT(m_solver.info() == Eigen::Success, "ERROR: system matrix decomposition failed");
        t.start();
        vec(sol) = m_solver.solve(vec(rhs));
        time = t.click(false).count();
        ZIRAN_INFO("Solving linear system takes ", time, "s");
    }

    void buildDiagonal(int opt = 0)
    {
        diagonalVal.resize(entryVal.size() / colsize);
        tbb::parallel_for(0, (int)diagonalVal.size(), [&](int i) {
            //for (int i = 0; i < (int)diagonalVal.size(); ++i) {
            TM d = TM::Zero();
            for (int idx = i * colsize; idx < (i + 1) * colsize; ++idx)
                if (entryCol[idx] == i) {
                    d += entryVal[idx];
                }
            diagonalVal[i] = d;
            //}
        });
        if (opt == 0) {
            diagonalEntry.resize(diagonalVal.size());
            tbb::parallel_for(0, (int)diagonalVal.size(), [&](int i) {
                diagonalEntry[i] = diagonalVal[i].diagonal().asDiagonal().inverse();
            });
        }
        diagonalBlock.resize(diagonalVal.size());
        tbb::parallel_for(0, (int)diagonalVal.size(), [&](int i) {
            diagonalBlock[i] = diagonalVal[i].inverse();
        });
    }

    T innerP(const TVStack& ddv, const TVStack& residual) const
    {
        ZIRAN_ASSERT(residual.cols() == ddv.cols());
        T result = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, residual.cols(), 256), (T)0,
            [&](const tbb::blocked_range<int>& range, T ns) -> T {
                int start = range.begin();
                int length = (range.end() - range.begin());
                const auto& ddv_block = ddv.middleCols(start, length);
                const auto& r_block = residual.middleCols(start, length);
                T to_add = ((ddv_block.transpose() * r_block).diagonal().array()).sum();
                return ns += to_add;
            },
            [](T a, T b) -> T { return a + b; });
        return result;
    }

    T extremeEigVal(std::string tag)
    {
        using ColSpMat = Eigen::SparseMatrix<T, Eigen::ColMajor>;
        using ColSpIter = typename ColSpMat::InnerIterator;
        std::vector<Eigen::Triplet<T>> triplets;
        int rowcnt = rows();
        for (int i = 0; i < rowcnt; ++i) {
            int st = i * colsize;
            int ed = st + colsize;
            for (; st < ed; ++st) {
                int j = entryCol[st];
                TM val = entryVal[st];
                for (int a = 0; a < dim; ++a)
                    for (int b = 0; b < dim; ++b) {
                        if (val(a, b) != 0)
                            triplets.emplace_back(i * dim + a, j * dim + b, val(a, b));
                    }
            }
        }
        ColSpMat mat;
        mat.resize(rowcnt * dim, rowcnt * dim);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        typedef Eigen::SimplicialLDLT<ColSpMat> SparseChol;
        typedef Eigen::ArpackGeneralizedSelfAdjointEigenSolver<ColSpMat, SparseChol> Arpack;
        Arpack arpack;
        int nbrEigenvalues = 1;
        arpack.compute(mat, nbrEigenvalues, tag.data());
        std::cout << "arpack eigenvalues of system matrix\n"
                  << arpack.eigenvalues().transpose() << std::endl;
        return arpack.eigenvalues()(0);
    }

    void estimate2norm(T tol = 1e-6)
    {
        constexpr int MaxIters = 512;
        int rowcnt = rows();

        //lMax = 15;
        //lMin = lMax / 30; ///< experience
        //return;
#if 0
        auto FNorm = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, rowcnt, 256), (T)0,
            [&](const tbb::blocked_range<int>& range, T ns) -> T {
                T localSum = 0;
                for (int i = range.begin(); i < range.end(); ++i) {
                    for (int st = i * colsize, ed = st + colsize; st < ed; ++st) {
                        localSum += entryVal[st].squaredNorm();
                    }
                }
                return ns + localSum;
            },
            [](T a, T b) -> T { return a + b; });
        lMax = std::sqrt(FNorm);
        lMin = lMax / 30; ///< experience
        return;
#endif

        TVStack v, x;
        v.resize(dim, rowcnt);
        x.resize(dim, rowcnt);
        auto setRandom = [](TVStack& v, bool normalize = false) {
            srand(time(NULL));
            tbb::parallel_for(0, (int)v.cols(), [&](int i) {
                for (int d = 0; d < dim; ++d) {
                    v(d, i) = rand() & 0x7;
                    if (normalize) {
                        if (v(d, i) >= 4)
                            v(d, i) = 1;
                        else
                            v(d, i) = -1;
                    }
                }
            });
        };

        setRandom(v, true);
#if 0
        int iter = 0;
        T e = 0, e0;
        do {
            e0 = e;
            multiply(v, x);
            T vv = innerP(v, v);
            T vAv = innerP(v, x);
            e = std::abs(vAv / vv);
            v = x / x.norm();
            iter++;
        } while (iter < MaxIters && std::abs(e - e0) > tol * e);
#else
        multiply(v, x);
        x = x.array().abs();
        T e = x.norm();
        if (e == 0) {
            lMin = lMax = 0;
            return;
        }
        x /= e;
        T e0 = 0;
        auto allZero = [tol](const TVStack& x) -> bool {
            T a = tol; // on Mac every variable in the [] must be used...
            bool allZero = true;
            tbb::parallel_for(0, (int)x.cols(), [&](int i) {
                for (int d = 0; d < dim; ++d)
                    if (x(d, i) != 0) {
                        allZero = false;
                        return;
                    }
            });
            return allZero;
        };
        int iter = 0;
        for (; iter < MaxIters && std::abs(e - e0) > tol * e; iter++) {
            e0 = e;
            multiply(x, v);
            if (allZero(v))
                v.setRandom();
            multiply(v, x);
            T normx = x.norm();
            e = normx / v.norm();
            x /= normx;
        }
        //T exactVal = extremeEigVal("LA");
        //T minVal = extremeEigVal("SA");
#endif
        if (iter >= MaxIters)
            ZIRAN_WARN("2Norm estimate of the system matrix might not be accurate!");
        ZIRAN_INFO("Estimated matrix 2norm after ", iter, " iterations is ", e /*, ", the exact one is ", exactVal*/);
        //ZIRAN_INFO("Min eig val is ", minVal);
        lMax = e;
        lMin = lMax / 30; ///< experience
        //lMin = minVal; ///< experience
    }

    void multiply(const TVStack& x, TVStack& b) const
    {
        ZIRAN_ASSERT((int)entryCol.size() / colsize == b.cols(), "Dims goes wrong :", (int)entryCol.size() / colsize, " ", x.cols());
        tbb::parallel_for(0, (int)b.cols(), [&](int i) {
            TV sum = TV::Zero();
            for (int idx = i * colsize; idx < (i + 1) * colsize; ++idx) {
                sum += entryVal[idx] * x.col(entryCol[idx]);
            }
            b.col(i) = sum;
        });
    }

    void transferAverage(const Vec& x, Vec& b) const
    {
        ZIRAN_ASSERT((int)entryCol.size() / colsize == b.size(), "Dims goes wrong :", (int)entryCol.size() / colsize, " ", x.size());
        tbb::parallel_for(0, (int)b.size(), [&](int i) {
            T sum = 0, wsum = 0;
            for (int idx = i * colsize; idx < (i + 1) * colsize; ++idx) {
                wsum += entryVal[idx](0, 0);
                sum += entryVal[idx](0, 0) * x(entryCol[idx]);
            }
            b(i) = sum / wsum;
        });
    }

    void transferMassWeightedSum(const Vec& x, const Vec& m, Vec& b) const
    {
        ZIRAN_ASSERT((int)entryCol.size() / colsize == b.size(), "Dims goes wrong :", (int)entryCol.size() / colsize, " ", x.size());
        tbb::parallel_for(0, (int)b.size(), [&](int i) {
            T sum = 0;
            for (int idx = i * colsize; idx < (i + 1) * colsize; ++idx) {
                sum += entryVal[idx](0, 0) * m(entryCol[idx]) * x(entryCol[idx]);
            }
            b(i) = sum;
        });
    }

    void transferWeightedSum(const Vec& x, Vec& b) const
    {
        ZIRAN_ASSERT((int)entryCol.size() / colsize == b.size(), "Dims goes wrong :", (int)entryCol.size() / colsize, " ", x.size());
        tbb::parallel_for(0, (int)b.size(), [&](int i) {
            T sum = 0;
            for (int idx = i * colsize; idx < (i + 1) * colsize; ++idx) {
                sum += entryVal[idx](0, 0) * x(entryCol[idx]);
            }
            b(i) = sum;
        });
    }

    void buildCoarseMatrix(const SquareMatrix<T, dim>& l, const SquareMatrix<T, dim>& r)
    {
        // get colsize
        colsize = 0;
        std::mutex mtx;
        tbb::parallel_for(0, (int)l.entryCol.size() / l.colsize, [&](int i) {
            std::unordered_map<int, bool> mp;
            for (int j = i * l.colsize; j < (i + 1) * l.colsize; ++j) {
                int jj = l.entryCol[j];
                for (int k = jj * r.colsize; k < (jj + 1) * r.colsize; ++k) {
                    int kk = r.entryCol[k];
                    mp[kk] = true;
                }
            }
            mtx.lock();
            colsize = std::max(colsize, (int)mp.size());
            mtx.unlock();
        });

        entryCol.resize((l.entryCol.size() / l.colsize) * colsize);
        entryVal.resize((l.entryCol.size() / l.colsize) * colsize);
        tbb::parallel_for(0, (int)l.entryCol.size() / l.colsize, [&](int i) {
            std::unordered_map<int, TM> mp;
            for (int j = i * l.colsize; j < (i + 1) * l.colsize; ++j) {
                int jj = l.entryCol[j];
                for (int k = jj * r.colsize; k < (jj + 1) * r.colsize; ++k) {
                    int kk = r.entryCol[k];
                    if (mp.find(kk) == mp.end()) {
                        mp[kk] = TM::Zero();
                    }
                    mp[kk] += l.entryVal[j] * r.entryVal[k];
                }
            }
            int idx = i * colsize;
            for (const auto& p : mp) {
                entryCol[idx] = p.first;
                entryVal[idx] = p.second;
                idx++;
            }
            for (; idx < (i + 1) * colsize; ++idx) {
                //entryCol[idx] = entryCol[i * colsize];
                entryCol[idx] = i > 0 ? 0 : 1;
                entryVal[idx] = TM::Zero();
            }
        });
    }

    void buildTransposeMatrix(const SquareMatrix<T, dim>& l, int rowcnt)
    {
        std::vector<std::unordered_map<int, TM>> data;
        data.resize(rowcnt);
        //tbb::parallel_for(0, (int)l.entryCol.size() / l.colsize, [&](int i) {
        for (int i = 0; i < (int)l.entryCol.size() / l.colsize; ++i) /// if parallelize, put a lock on data's first dimension
            for (int j = i * l.colsize; j < (i + 1) * l.colsize; ++j) {
                int jj = l.entryCol[j];
                if (data[jj].find(i) == data[jj].end())
                    data[jj][i] = TM::Zero();
                data[jj][i] += l.entryVal[j];
            }
        //});
        colsize = 0;
        for (int i = 0; i < (int)data.size(); ++i) {
            colsize = std::max(colsize, (int)data[i].size());
        }
        entryCol.resize(data.size() * colsize);
        entryVal.resize(data.size() * colsize);
        tbb::parallel_for(0, (int)data.size(), [&](int i) {
            //for (int i = 0; i < (int)data.size(); ++i) {
            int idx = i * colsize;
            for (const auto& p : data[i]) {
                entryCol[idx] = p.first;
                entryVal[idx] = p.second;
                idx++;
            }
            for (; idx < (i + 1) * colsize; ++idx) {
                //entryCol[idx] = entryCol[i * colsize];
                entryCol[idx] = i > 0 ? 0 : 1;
                entryVal[idx] = TM::Zero();
            }
        });
        //}
    }

    std::vector<TM> testValue()
    {
        std::vector<TM> result;
        for (int i = 0; i < (int)entryVal.size() / colsize; ++i) {
            TM tmp = TM::Zero();
            for (int idx = i * colsize; idx < (i + 1) * colsize; ++idx) {
                tmp += entryVal[idx] * (T)entryCol[idx];
            }
            result.push_back(tmp);
        }
        return result;
    }
};

}; // namespace ZIRAN

#endif