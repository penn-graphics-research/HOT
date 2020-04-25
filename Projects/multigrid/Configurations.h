#ifndef __CONFIGURATIONS_H_
#define __CONFIGURATIONS_H_

#include <string>
#include <vector>

namespace CmdArgument {
/// cmd0 for the hard young's modulus, cmd1 for the soft young's modulus
inline static double cmd0 = 0, cmd1 = 0;
} // namespace CmdArgument

namespace CaseSettings {

inline static double v_mu = 0.001;
inline static std::string output_folder;
} // namespace CaseSettings

namespace HOTSettings {
inline static double cgratio = 0.5;
inline static double characterNorm;
inline static double cneps{ 1e-5 };
inline static bool useAdaptiveHessian{ false };
inline static bool useCN{ false };
inline static bool matrixFree{ false }; ///< 0: matrix free, 1: matrix
inline static bool project{ false }; ///< project dPdF or not
inline static bool systemBCProject{ false }; ///< BC project system (matrix & rhs) or not
inline static bool linesearch{ false };
inline static int boundaryType{ 0 }; ///< 0: all sticky 1: has slip
inline static int lsolver{ 0 }; ///< 0: newton with multigrid solver 1: newton with minres 2: newton with cg 3: LBFGS with multigrid 4: direct solver
inline static int Ainv{ 0 }; ///< 0: entry 1: block 2: mass (only works with minres)
inline static int smoother{ 0 }; ///< 0: (optimal) Jacobi, 1: Optimal Jacobi, 2: PCG, 3: LBFGS-OJ, 4: Minres, 5: GS, 6: Chebyshev, 7: Incomplete Cholesky
inline static int coarseSolver{ 0 }; ///< Options same with smoother
inline static int levelCnt{ 1 }, times{ 1 };
inline static int levelscale{ 0 };
inline static int debugMode{ 0 }; ///< open all
inline static double omega{ 1 };
inline static double topomega{ 0.1 };
inline static bool revealJacobi{ false };
inline static bool revealVcycle{ false };
inline static bool topDownMGS{ false };
inline static bool useBaselineMultigrid{ false };
} // namespace HOTSettings

#define USE_GAST15_METRIC 0

#endif