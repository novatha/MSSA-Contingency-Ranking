# ==============================================================================
# UNIFIED BENCHMARK RUNNER FOR POWER SYSTEM N-k CONTINGENCY RANKING
# Paper: "A Risk-Based Modified Salp Swarm Algorithm for N-k Contingency Ranking"
# Authors: Novalio Daratha, Fitra Akbar, Adhadi Kurniawan, Hendy Santosa
# ==============================================================================

using PowerModels
using Ipopt
using Printf

include("MSSA_direct_nk.jl")

function print_banner()
    println("================================================================================")
    println(" POWER SYSTEM N-k CONTINGENCY RANKING REPRODUCIBILITY SUITE")
    println(" Algorithm: Direct On-The-Fly Modified Salp Swarm Algorithm (MSSA)")
    println("================================================================================")
end

function main()
    print_banner()

    case = "case30"
    k = 2

    if length(ARGS) >= 1
        case = ARGS[1]
    end
    if length(ARGS) >= 2
        k = parse(Int, ARGS[2])
    end

    if case == "--all" || case == "all"
        println("\n>>> Running Full Benchmark Suite across Standard Test Cases...\n")
        cases = [
            ("case14", 1),
            ("case30", 2),
            ("case57", 2),
            ("case118", 2),
            ("case300", 2),
            ("casesumatera", 2),
            ("case30", 3),
            ("case57", 3)
        ]

        summary_results = []
        for (c, k_val) in cases
            println("\n------------------------------------------------------------")
            println(" Testing System: $c | Outage Order: N-$k_val")
            println("------------------------------------------------------------")
            archive, t_exec, evals = run_direct_mssa(c, k_val)
            push!(summary_results, (case=c, k=k_val, time=t_exec, evals=evals, top1_risk=archive[1].risk_index))
        end

        println("\n================================================================================")
        println("                           SUMMARY BENCHMARK RESULTS                            ")
        println("================================================================================")
        println(" System       | Contingency | Runtime (s) | Evaluated Combinations | Top-1 Risk ")
        println("--------------------------------------------------------------------------------")
        for res in summary_results
            println(@sprintf(" %-12s | N-%-9d | %11.4f | %22d | %10.4e",
                res.case, res.k, res.time, res.evals, res.top1_risk))
        end
        println("================================================================================")
    else
        run_direct_mssa(case, k)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
