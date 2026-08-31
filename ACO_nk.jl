# ACO_nk.jl - Ant Colony Optimization for N-k Contingency Analysis
# Based on the methodology in transaction_paper_v2.2.tex

using PowerModels
using Ipopt
using DelimitedFiles
using Statistics
using Combinatorics
using SpecialFunctions
using Random
using Distributed
using Base.Threads
using StatsBase # For weights in sampling

# Include configuration and utilities
include("config.jl")
include("contingency_utils.jl")
include("analisis_nk_optimized_final.jl")

# --- ACO Specific Functions ---

function initialize_aco(dim, values)
    # Initialize Pheromones
    # Small constant value, e.g., 1 / (Ng)
    tau0 = 1.0 / dim
    tau = fill(tau0, dim)
    
    # Heuristic Information (Risk Index itself)
    # Add small epsilon to avoid 0
    eta = values .+ 1e-10
    
    return tau, eta
end

function construct_solution(tau, eta, alpha, beta, dim, capacity)
    # Ants construct solution probabilistically
    # Probability P_j = (tau_j^alpha * eta_j^beta) / sum(...)
    
    # Pre-calculate numerators (static for this ant's full construction step)
    # Note: In highly dynamic ACO, this updates step-by-step if tau updates online.
    # Here we assume batch update (Ant System / MMAS).
    
    numerators = (tau .^ alpha) .* (eta .^ beta)
    
    # Initialize empty solution
    solution = zeros(Int, dim)
    current_weight = 0
    
    # Available items (indices)
    available_indices = collect(1:dim)
    
    # Efficient Sampling:
    # Instead of re-normalizing every single step for all items, 
    # we can use StatsBase.sample with weights.
    # But weights change as we remove items. 
    
    # For Knapsack with capacity ~50% of N (~1000+ items), 
    # loop is expensive. 
    
    # Optimization: Select 'capacity' items.
    # If we use 'sample(available, Weights(nums), capacity, replace=false)'
    # it is efficiently implemented in Julia.
    
    selected_indices = sample(available_indices, Weights(numerators), capacity, replace=false)
    
    solution[selected_indices] .= 1
    
    return solution
end

function run_aco_loop(tau, eta, alpha, beta, rho, weights, values, capacity, dim, N)
    GlobalBestPos = zeros(Int, dim)
    GlobalBestFit = -Inf
    
    Convergence_curve = zeros(Max_iter)
    
    l = 1
    no_improvement_count = 0
    
    sorted_indices_for_opt_fitness = sortperm(values, rev=true)
    manual_indices_for_opt_fitness = sorted_indices_for_opt_fitness[1:min(20, length(values))]
    optimal_fitness_value = sum(values[manual_indices_for_opt_fitness])
    
    while l <= Max_iter && GlobalBestFit < optimal_fitness_value && no_improvement_count < 100
        
        # Solution Construction Phase (Parallelizable?)
        # Since ants are independent in this phase (batch update), we can thread.
        
        # Solutions storage
        AntSolutions = zeros(Int, N, dim)
        AntFitnesses = zeros(Float64, N)
        
        # NOTE: rand() is not thread-safe in older Julia versions without care, 
        # but modern Julia threads usually handle separate RNG states if using `rand()`.
        # `sample` from StatsBase might need care.
        # For safety and simplicity in this version, we run serial or lock-free if possible.
        # Given ACO is slow, let's try to keep it correct first.
        
        numerators = (tau .^ alpha) .* (eta .^ beta)
        
        # Threads.@threads for i in 1:N 
        # Using threads with weighted sampling might be tricky with shared 'numerators'.
        # 'numerators' is read-only here, so it should be fine.
        
        for i in 1:N
            # Select items
            # We select 'capacity' items
            selected_indices = sample(1:dim, Weights(numerators), capacity, replace=false)
            
            AntSolutions[i, selected_indices] .= 1
            
            # Evaluate
            # Simple sum since weights are 1 and we enforced capacity exactly
            AntFitnesses[i] = sum(values[selected_indices])
        end
        
        # Find Iteration Best
        iter_best_val, iter_best_idx = findmax(AntFitnesses)
        iter_best_pos = AntSolutions[iter_best_idx, :]
        
        # Update Global Best
        if iter_best_val > GlobalBestFit
            GlobalBestFit = iter_best_val
            GlobalBestPos = iter_best_pos
            no_improvement_count = 0
        else
            no_improvement_count += 1
        end
        
        # Pheromone Update
        # 1. Evaporation
        tau .*= (1.0 - rho)
        
        # 2. Deposit (Global Best - Elitist Strategy)
        # Amount to deposit? standard is Q / L or just proportional.
        # For maximization, we can add constant * Fitness, or just Fitness.
        # To prevent explosion, normalize?
        # Let's use: tau_new = tau_old + rho * Fitness (MMAS style roughly)
        # Or simply: tau[j] += GlobalBestFit if j in GlobalBest
        
        # Check "transaction_paper" methodology implied standard ACO.
        # We'll use: Add proportional to fitness for Best Ant.
        
        deposit_indices = findall(x -> x == 1, GlobalBestPos)
        # Scale deposit to be meaningful relative to tau0 (~1/Ng)
        # If fitness is large (e.g. 100.0), and tau0 is 0.001.
        # Deposit 0.1 * (Fitness / MaxFitnessEstimate)?
        
        # Let's use a learning rate approach:
        # tau[j] = (1-rho)*tau[j] + rho * 1 (if in best)
        # This keeps tau in [0, 1] range roughly.
        
        for idx in deposit_indices
            tau[idx] += rho * 1.0 
        end
        
        Convergence_curve[l] = GlobalBestFit
        l += 1
    end
    
    return GlobalBestPos, GlobalBestFit, Convergence_curve, l
end

function run_local_search_aco(GlobalBestPos, GlobalBestFit, weights, values, capacity, dim)
    # Reuse standard hill climbing
    current_best_solution_ls = copy(GlobalBestPos)
    current_best_fitness_ls = GlobalBestFit
    improved_ls = true
    
    while improved_ls
        improved_ls = false
        @inbounds @simd for j in 1:dim
            test_pos_ls = copy(current_best_solution_ls)
            test_pos_ls[j] = 1 - test_pos_ls[j]
            
            # REPAIR is crucial here because flipping 0->1 might violate capacity
            test_pos_ls = repair_knapsack_solution(test_pos_ls, weights, values, capacity)
            
            test_fitness_ls = 0.0
            @inbounds @simd for k_idx in 1:dim
                if test_pos_ls[k_idx] == 1
                    test_fitness_ls += values[k_idx]
                end
            end
            
            if test_fitness_ls > current_best_fitness_ls
                current_best_fitness_ls = test_fitness_ls
                current_best_solution_ls = test_pos_ls
                improved_ls = true
            end
        end
    end
    return current_best_solution_ls, current_best_fitness_ls
end

# --- Main Function ---

function ACO_nk(case_name::String, k::Int=2)
    println("=== ACO N-", k, " CONTINGENCY ANALYSIS for ", case_name, " ===")
    t_analisis = time()
    mpc = PowerModels.parse_file(case_name)

    contingencies_nk = generate_nk_contingencies(mpc, k)
    Ng = size(contingencies_nk, 1)
    println("Number of N-", k, " contingencies: ", Ng)

    println("Performing exhaustive analysis for fitness landscape...")
    IP_V_SELURUH = Vector{Float64}(undef, Ng)
    Threads.@threads for i in 1:Ng
        IP_V_SELURUH[i] = analisis_nk(mpc, contingencies_nk[i, :])
    end
    
    t_analisis_total = time() - t_analisis
    println("Time for N-", k, " analysis (s)            : ", (round(t_analisis_total, digits=4)))

    contingency_probabilities = Vector{Float64}(undef, Ng)
    @inbounds for i in 1:Ng
        current_nk_contingency = contingencies_nk[i, :]
        joint_probability = 1.0
        @inbounds @simd for j in 1:length(current_nk_contingency)
            branch_idx = current_nk_contingency[j]
            joint_probability *= get_contingency_probability_single(mpc, branch_idx)
        end
        contingency_probabilities[i] = joint_probability
    end

    # Parameters
    weights = ones(Ng)
    values = IP_V_SELURUH .* contingency_probabilities
    capacity = Int(round(0.5 * Ng))
    dim = Ng
    
    # ACO Parameters
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    
    CR_all = zeros(Nrun)
    Time_all = zeros(Nrun)

    t_start = time()
    for runIdx in 1:Nrun
        println("Running ACO ke-", runIdx)
        t_run = time()

        # Initialize
        tau, eta = initialize_aco(dim, values)
        
        # Loop
        GBestPos, GBestFit, Convergence_curve, l = run_aco_loop(tau, eta, alpha, beta, rho, weights, values, capacity, dim, N)
        
        # Hybrid Local Search
        GBestPos, GBestFit = run_local_search_aco(GBestPos, GBestFit, weights, values, capacity, dim)
        
        # Metrics
        manual_idx = sortperm(values, rev=true)
        manual_indices_top20 = manual_idx[1:min(20, length(values))]
        
        selected_idx = findall(x -> x == 1, GBestPos)
        sort_order = sortperm(values[selected_idx], rev=true)
        aco_indices_top20 = selected_idx[sort_order[1:min(20, length(sort_order))]]
        
        Cr = calculate_capture_ratio(manual_indices_top20, aco_indices_top20)
        
        CR_all[runIdx] = Cr
        Time_all[runIdx] = time() - t_run
        
        open("results_aco_run_$(runIdx).txt", "w") do f
            writedlm(f, GBestPos)
            writedlm(f, [GBestFit])
        end
    end
    
    t_total = time() - t_start
    avgTime = mean(Time_all)

    println(" ")
    println("=== RATA-RATA HASIL $Nrun RUNNING (ACO) ===")
    println("Time for exhaustive analysis (s)             : ", (round(t_analisis_total, digits=4)))
    println("Average Time per ACO run (s)                 : ", (round(avgTime, digits=4)))
    println("Rata-rata Capture Ratio (%)"                  , ": ", (round(mean(CR_all), digits=2)))
    println("Std Capture Ratio (%)"                        , ": ", (round(std(CR_all), digits=2)))
    println("Min Capture Ratio (%)"                        , ": ", (round(minimum(CR_all), digits=2)))
    println("Max Capture Ratio (%)"                        , ": ", (round(maximum(CR_all), digits=2)))
    
    println("\n=============================================================")
    println("SUCCESS: ACO completed for ", case_name)
    println("=============================================================")
end

# Execute if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    ACO_nk("case30.m", 2)
end
