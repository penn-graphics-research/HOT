
cnepss = [1e-4, 1e-5, 1e-6, 1e-7]

function generate_instrs()
    io = open("/home/mine/Codes/ziran/Projects/multigrid/benchmark_script.sh", "w")
    # vanilla newton vs. proj newton
    # proj?, solver, case
    for i in 26:28
    for eps in cnepss
    for lsolver in 1:2
    for proj in 0:1
        println(io, "./multigrid -test $i --3d --usecn -cneps $eps -lsolver $lsolver -Ainv 2")
    end
    end
    end
    end
    # matrix vs. matrix-free
    close(io)
end

generate_instrs()

# pn with mg as preconditioner
# smoother only jacobi or strict gauss-seidel
# linesearch
levels = [1, 2, 3, 4, 5, 6]
times = [1, 2, 4, 8, 16]
omegas = [0.05, 0.1, 0.2, 0.3, 1]   # last one only for GS
scales = [0, 1, 2, 4, 6]
omegas = [0.05, 0.1, 0.2, 0.3, 1]   # last one only for GS

# lbfgs-amg
smoothers = [0, 1, 2, 3, 4, 5, 6]
