#!/bin/bash
# gast15, remember to set USE_GAST15_METRIC to 1
declare -a StringArray1=("1e-3")
for val in ${StringArray1[@]}; do
./multigrid -test 888999111 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --matfree -o box_pnmf_gast15_tol$val
./multigrid -test 888999000 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --matfree -o donutNocandy_pnmf_gast15_tol$val
./multigrid -test 777017 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --matfree -o boards_pnmf_gast15_tol$val
./multigrid -test 777010 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --matfree -o chain_pnmf_gast15_tol$val
./multigrid -test 777014 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --matfree -o fracture_pnmf_gast15_tol$val
./multigrid -test 777019 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --matfree -o wheel_pnmf_gast15_tol$val
./multigrid -test 777011 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --matfree -o faceless_pnmf_gast15_tol$val
./multigrid -test 9212 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch -cmd0 1e9 --matfree -o armacatstiff_gast15_pnmf_tol$val
./multigrid -test 9212 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch -cmd0 1e6 --matfree -o armacatsoft_gast15_pnmf_tol$val
./multigrid -test 777001 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch -cmd0 1e9 --matfree -o twistbar_pnmf-1e9-gast15-tol$val
./multigrid -test 777020 --3d -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --matfree -o turkey_pnmf_gast15_tol$val
done
./multigrid -test 777020 --3d --usecn -cneps 1e2 -lsolver 2 -Ainv 1 --project --linesearch --matfree -o turkey_pnmf_gast15_tol_1e2

# pnpcg, pnpcg(mf)
declare -a StringArray1=("1e-7")
declare -a StringArray2=("777014" "888999111" "888999000" "777010" "777017" "777019" "777011" "777020")
for val in ${StringArray1[@]}; do
    for case in ${StringArray2[@]}; do
        ./multigrid -test $case --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch -o $case_pn_tol$val
        ./multigrid -test $case --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --matfree -o $case_pnmf_tol$val
    done
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch -cmd0 1e9 -o armacatstiff_pn_tol$val
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch -cmd0 1e9 --matfree -o armacatstiff_pnmf_tol$val
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch -cmd0 1e6 -o armacatsoft_pn_tol$val
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch -cmd0 1e6 --matfree -o armacatsoft_pnmf_tol$val
done

# HOT
declare -a StringArray1=("1e-7")
declare -a StringArray2=("777014" "888999111" "888999000" "777010" "777017" "777019" "777011" "777020")
for val in ${StringArray1[@]}; do
    for case in ${StringArray2[@]}; do
        ./multigrid -test $case --3d --usecn -cneps $val -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 -o $case_HOT_tol$val
    done
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 -cmd0 1e9 -o armacatstiff_HOT_tol$val
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 -cmd0 1e6 -o armacatsoft_HOT_tol$val
done

# pn-mgpcg
declare -a StringArray1=("1e-7")
declare -a StringArray2=("777014" "888999111" "888999000" "777010" "777017" "777019" "777011" "777020")
for val in ${StringArray1[@]}; do
    for case in ${StringArray2[@]}; do
        ./multigrid -test $case --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 -o $case_pnmgpcg_tol$val
    done
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 -cmd0 1e9 -o armacatstiff_pnmgpcg_tol$val
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 -cmd0 1e6 -o armacatsoft_pnmgpcg_tol$val
done

# lbfgs-H
declare -a StringArray1=("1e-7")
declare -a StringArray2=("777014" "888999111" "888999000" "777010" "777017" "777019" "777011" "777020")
for val in ${StringArray1[@]}; do
    for case in ${StringArray2[@]}; do
        ./multigrid -test $case --3d --usecn -cneps $val -lsolver 3 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 10000 -coarseSolver 2 -smoother 2 -o $case_lbfgsH_tol$val
    done
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 3 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 10000 -coarseSolver 2 -smoother 2 -cmd0 1e9 -o armacatstiff_lbfgsH_tol$val
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 3 -Ainv 1 --project --linesearch -mg_level 1 -mg_times 10000 -coarseSolver 2 -smoother 2 -cmd0 1e6 -o armacatsoft_lbfgsH_tol$val
done

# lbfgs-GMG, pn-GMG-pcg
declare -a StringArray1=("1e-7")
declare -a StringArray2=("777014" "888999111" "888999000" "777010" "777017" "777019" "777011" "777020")
for val in ${StringArray1[@]}; do
    for case in ${StringArray2[@]}; do
        ./multigrid -test $case --3d --usecn -cneps $val -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 --baseline -o $case_lbfgsGMG_tol$val
        ./multigrid -test $case --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 --baseline -o $case_pnGMGpcg_tol$val
    done
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 --baseline -cmd0 1e9 -o armacatstiff_lbfgsGMG_tol$val
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 3 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 --baseline -cmd0 1e6 -o armacatsoft_lbfgsGMG_tol$val
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 --baseline -cmd0 1e9 -o armacatstiff_pnGMGpcg_tol$val
    ./multigrid -test 9212 --3d --usecn -cneps $val -lsolver 2 -Ainv 1 --project --linesearch --bcproject -mg_level 3 -mg_times 1 -coarseSolver 2 -smoother 5 --baseline -cmd0 1e6 -o armacatsoft_pnGMGpcg_tol$val
done
