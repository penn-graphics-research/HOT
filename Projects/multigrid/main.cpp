#include <Ziran/CS/Util/FloatingPointExceptions.h>
#include <Ziran/CS/Util/SignalHandler.h>
#include <Ziran/CS/Util/CommandLineFlags.h>
#include <Ziran/CS/Util/Debug.h>
#include <Ziran/CS/Util/Filesystem.h>
#include <Ziran/CS/Util/PluginManager.h>
#include <tbb/tbb.h>
#include "MultigridInit2D.h"
#include "MultigridInit3D.h"
#include "Configurations.h"

#define T double
#define dim 3

using namespace ZIRAN;

int main(int argc, char* argv[])
{
    // std::vector<int> entryCol;
    // std::vector<T> entryVal;
    // SparseMatrix<T, dim> mtr(entryCol, entryVal);

    {
        bool displayHelp = false;
        int test_number = -1; // Non-lua option.
        bool three_d = false; // Non-lua option.
        bool use_double = false; // Non-lua option
        int restart = 0; // Non-lua option
        bool run_diff_test = false; // Non-lua option
        double diff_test_perturbation_scale = 1; // Non-lua option

        // Not checking for nan, because when constitutive model returns that, MpmForceBase is skipping them (treating as zeros)
        // FPE::WatchedScope w(FPE::Mask::Overflow | FPE::Mask::DivZero);
        // Unconmment the following to catch division by 0
        // FPE::WatchedScope w(FPE::Mask::Overflow | FPE::Mask::Invalid | FPE::Mask::DivZero);
        FPE::WatchedScope w(FPE::Mask::Invalid);

        std::string script_file_name;
        StdVector<std::string> inline_strings;
        FLAGS::Register helpflag("--help", "Print help (this message) and exit", displayHelp);

        // Lua command line options
        FLAGS::Register scriptflag("-script", "Lua script to read for initial data", script_file_name);
        FLAGS::Register iflag("-i", "Append string to script", inline_strings);

        // Non-lua command line options
        FLAGS::Register test_number_flag("-test", "Test number (non-lua test)", test_number);
        FLAGS::Register three_d_flag("--3d", "Dimension is 3(non-lua test)", three_d);
        FLAGS::Register run_diff_test_flag("--run_diff_test", "Run diff test (non-lua test)", run_diff_test);
        FLAGS::Register diff_test_perturbation_scale_flag("-dtps", "diff_test_perturbation_scale (non-lua test)", diff_test_perturbation_scale);
        FLAGS::Register double_flag("--double", "Dimension (non-lua test)", use_double);
        FLAGS::Register restart_flag("-restart", "Restart frame (non-lua test)", restart);

        FLAGS::Register v_mu_flag("-v_mu", "v_mu", CaseSettings::v_mu);
        FLAGS::Register output_folder_flag("-o", "output_folder overwrite", CaseSettings::output_folder);
        FLAGS::Register dbg_flag("-dbg", "debug mode (0: close 1: open)", HOTSettings::debugMode);
        FLAGS::Register cneps_flag("-cneps", "epsilon for characteristic norm", HOTSettings::cneps);
        FLAGS::Register usel_baselinemg_flag("--baseline", "if use baseline multigrid", HOTSettings::useBaselineMultigrid);
        FLAGS::Register usecn_flag("--usecn", "if use characteristic norm", HOTSettings::useCN);
        FLAGS::Register use_adaptiveH_flag("--adaptiveH", "if use adaptive hessian", HOTSettings::useAdaptiveHessian);
        FLAGS::Register matrix_flag("--matfree", "if matrix-free", HOTSettings::matrixFree);
        FLAGS::Register proj_flag("--project", "if project matrix", HOTSettings::project);
        FLAGS::Register bc_proj_flag("--bcproject", "if project boundary on matrix", HOTSettings::systemBCProject);
        FLAGS::Register linesearch_flag("--linesearch", "if using linesearch", HOTSettings::linesearch);
        FLAGS::Register boundary_flag("-bc", "boundary condition type", HOTSettings::boundaryType);
        FLAGS::Register lsolver_flag("-lsolver", "linear solver type", HOTSettings::lsolver);
        FLAGS::Register Ainv_flag("-Ainv", "A~ inverse", HOTSettings::Ainv);
        FLAGS::Register smoother_flag("-smoother", "smoother type", HOTSettings::smoother);
        FLAGS::Register coarseSolver_flag("-coarseSolver", "coarse solver type", HOTSettings::coarseSolver);
        FLAGS::Register mglevel_flag("-mg_level", "multigrid level", HOTSettings::levelCnt);
        FLAGS::Register mgtimes_flag("-mg_times", "smoother times", HOTSettings::times);
        FLAGS::Register levelscale_flag("-mg_scale", "multigrid smoother time scale", HOTSettings::levelscale);
        FLAGS::Register omega_flag("-mg_omega", "Gauss Seidel omega", HOTSettings::omega);
        FLAGS::Register jomega_flag("-mg_jomega", "Jacobi omega", HOTSettings::topomega);
        FLAGS::Register reveal_residual_flag("--showresidual", "multigrid residual time", HOTSettings::revealJacobi);
        FLAGS::Register reveal_vcycle_flag("--showvcycle", "multigrid vcycle time", HOTSettings::revealVcycle);
        FLAGS::Register topDownMGS_flag("--topDownMGS", "use top down MG solver", HOTSettings::topDownMGS);

        FLAGS::Register cmd0_flag("-cmd0", "cmd0", CmdArgument::cmd0);
        FLAGS::Register cmd1_flag("-cmd1", "cmd1", CmdArgument::cmd1);

        int num_threads = tbb::task_scheduler_init::automatic;

        FLAGS::Register thread_flag("-t", "Set number of threads", num_threads);

        std::stringstream script;
        PluginManager pm;
        try {
            FLAGS::ParseFlags(argc, argv);
            if (!script_file_name.empty()) {
                FILESYSTEM::readFile(script, script_file_name);
            }
            pm.loadAllPlugins();
        }
        catch (std::exception& e) {
            std::cerr << e.what() << std::endl;
            FLAGS::PrintUsage(std::cerr);
            return 1;
        }
        if (displayHelp) {
            std::cout << "Usage:\n";
            FLAGS::PrintUsage(std::cout);
            return 0;
        }
        installSignalHandler();
        for (const auto& s : inline_strings) {
            script << s << "\n";
        }

        tbb::task_scheduler_init init(num_threads);

        if (script_file_name.empty()) {
            ZIRAN_ASSERT(test_number != -1, "No lua script loaded. Either load with --script or set --test");
            MultigridSimulation<T, dim> e;
            if (run_diff_test) {
                e.diff_test = true;
                e.diff_test_perturbation_scale = diff_test_perturbation_scale;
            }
            e.logger = LogWorker::initializeLogging();
#if dim == 2
            MultigridInit2D<T> h(e, test_number);
#else
            MultigridInit3D<T> h(e, test_number);
#endif
            if (!restart)
                h.start();
            else
                h.restart(restart);
        }
    }
    return 0;
}
