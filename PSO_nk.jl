# PSO_nk.jl - Particle Swarm Optimization for N-k Contingency Analysis
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

# Include configuration and utilities
include("config.jl")
include("contingency_utils.jl")
include("analisis_nk_optimized_final.jl")

# --- PSO Specific Functions ---

function initialize_pso_population(N, dim, weights, values, capacity)
    # Initialize positions and velocities
    Positions = rand(0:1, N, dim)
    Velocities = zeros(Float64, N, dim) # Initial velocity 0
    
    PersonalBestPos = copy(Positions)
    PersonalBestFit = fill(-Inf, N)
    
    GlobalBestPos = zeros(Int, dim)
    GlobalBestFit = -Inf

    # Initial Evaluation
    @inbounds for i in 1:N
        pos = Positions[i, :]
        # Repair initial random solutions
        Positions[i, :] = repair_knapsack_solution(pos, weights, values, capacity)
        
        fitness = 0.0
        @inbounds @simd for j in 1:dim
            if Positions[i, j] == 1
                fitness += values[j]
            end
        end
        
        PersonalBestFit[i] = fitness
        PersonalBestPos[i, :] = Positions[i, :]
        
        if fitness > GlobalBestFit
            GlobalBestFit = fitness
            GlobalBestPos = copy(Positions[i, :])
        end
    end
    
    return Positions, Velocities, PersonalBestPos, PersonalBestFit, GlobalBestPos, GlobalBestFit
end

function run_pso_loop(Positions, Velocities, PersonalBestPos, PersonalBestFit, GlobalBestPos, GlobalBestFit, weights, values, capacity, dim, N)
    Convergence_curve = zeros(Max_iter)
    Convergence_curve[1] = GlobalBestFit
    
    # PSO Parameters
    w_max = 0.9
    w_min = 0.4
    c1 = 2.0
    c2 = 2.0
    v_max = 6.0 # Velocity clamping
    
    l = 2
    no_improvement_count = 0
    
    sorted_indices_for_opt_fitness = sortperm(values, rev=true)
    manual_indices_for_opt_fitness = sorted_indices_for_opt_fitness[1:min(20, length(values))]
    optimal_fitness_value = sum(values[manual_indices_for_opt_fitness])

    @inbounds while l <= Max_iter && GlobalBestFit < optimal_fitness_value && no_improvement_count < 100
        w = w_max - ((w_max - w_min) * l / Max_iter) # Inertia weight damping
        
        previous_global_best_fit = GlobalBestFit
        
        @inbounds for i in 1:N
            r1 = rand()
            r2 = rand()
            
            for j in 1:dim
                # Velocity Update
                Velocities[i, j] = w * Velocities[i, j] + 
                                   c1 * r1 * (PersonalBestPos[i, j] - Positions[i, j]) + 
                                   c2 * r2 * (GlobalBestPos[j] - Positions[i, j])
                
                # Clamp Velocity
                if Velocities[i, j] > v_max
                    Velocities[i, j] = v_max
                elseif Velocities[i, j] < -v_max
                    Velocities[i, j] = -v_max
                end
                
                # Sigmoid Transfer
                sigmoid = 1.0 / (1.0 + exp(-Velocities[i, j]))
                
                # Position Update
                if rand() < sigmoid
                    Positions[i, j] = 1
                else
                    Positions[i, j] = 0
                end
            end
            
            # Intelligent Repair (as per paper)
            Positions[i, :] = repair_knapsack_solution(Positions[i, :], weights, values, capacity)
            
            # Evaluation
            fitness = 0.0
            @inbounds @simd for j in 1:dim
                if Positions[i, j] == 1
                    fitness += values[j]
                end
            end
            
            # Update Personal Best
            if fitness > PersonalBestFit[i]
                PersonalBestFit[i] = fitness
                PersonalBestPos[i, :] = copy(Positions[i, :])
            end
            
            # Update Global Best
            if fitness > GlobalBestFit
                GlobalBestFit = fitness
                GlobalBestPos = copy(Positions[i, :])
            end
        end
        
        if GlobalBestFit > previous_global_best_fit
             no_improvement_count = 0
        else
             no_improvement_count += 1
        end
        
        Convergence_curve[l] = GlobalBestFit
        l += 1
    end
    
    return GlobalBestPos, GlobalBestFit, Convergence_curve, l
end

function run_local_search(GlobalBestPos, GlobalBestFit, weights, values, capacity, dim)
    # Hill Climbing Local Search (Same as MSSA)
    current_best_solution_ls = copy(GlobalBestPos)
    current_best_fitness_ls = GlobalBestFit
    improved_ls = true
    
    while improved_ls
        improved_ls = false
        @inbounds @simd for j in 1:dim
            test_pos_ls = copy(current_best_solution_ls)
            # Flip bit
            test_pos_ls[j] = 1 - test_pos_ls[j]
            
            # Repair and Evaluate
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

function PSO_nk(case_name::String, k::Int=2)
    println("=== PSO N-", k, " CONTINGENCY ANALYSIS for ", case_name, " ===")
    t_analisis = time()
    mpc = PowerModels.parse_file(case_name)

    contingencies_nk = generate_nk_contingencies(mpc, k)
    Ng = size(contingencies_nk, 1)
    println("Number of N-", k, " contingencies: ", Ng)

    # Pre-calculate IP_V and Probabilities (Simulating 'analisis_all' once)
    # In a real scenario, we might only evaluate needed ones, but for the 'benchmark' 
    # structure where we assume values are known or we want to compare optimization time
    # specifically, we often pre-calc or use a lookup.
    # However, the paper implies the algorithm CALLS the analysis. 
    # But standard comparison usually runs exhaustive FIRST to get ground truth (values),
    # then runs the metaheuristic using those values to find the max.
    # MSSA_nk_refactored_v2.jl calculates ALL IP_V first. We will follow that pattern.
    
    println("Performing exhaustive analysis for fitness landscape...")
    IP_V_SELURUH = Vector{Float64}(undef, Ng)
    # Parallelize exhaustive analysis
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

    # Knapsack Parameters
    weights = ones(Ng) # Weight is 1 per contingency
    values = IP_V_SELURUH .* contingency_probabilities # Risk Index
    capacity = Int(round(0.5 * Ng)) # 50% capacity as per paper (though typically we look for top 20)
    # Note: The paper mentions "Capacity = 50% of total contingencies" but we care about Top-20.
    # The knapsack formulation tries to select the 'best subset'. 
    # The Capture Ratio is calculated against the TRUE Top-20.
    
    dim = Ng
    
    CR_all = zeros(Nrun)
    Time_all = zeros(Nrun)

    t_start = time()
    @inbounds for runIdx in 1:Nrun
        println("Running PSO ke-", runIdx)
        t_run = time()

        Positions, Velocities, PBestPos, PBestFit, GBestPos, GBestFit = initialize_pso_population(N, dim, weights, values, capacity)
        
        GBestPos, GBestFit, Convergence_curve, l = run_pso_loop(Positions, Velocities, PBestPos, PBestFit, GBestPos, GBestFit, weights, values, capacity, dim, N)

        # Hybrid Local Search
        GBestPos, GBestFit = run_local_search(GBestPos, GBestFit, weights, values, capacity, dim)

        # Calculate Metrics (Capture Ratio)
        manual_idx = sortperm(values, rev=true)
        manual_indices_top20 = manual_idx[1:min(20, length(values))]
        
        selected_idx = findall(x -> x == 1, GBestPos)
        # If we selected more than 20 (due to 50% capacity), we take the top 20 OF THE SELECTED
        # based on their specific values.
        sort_order = sortperm(values[selected_idx], rev=true) # Sort selected by RI
        pso_indices_top20 = selected_idx[sort_order[1:min(20, length(sort_order))]]
        
        Cr = calculate_capture_ratio(manual_indices_top20, pso_indices_top20)
        
        CR_all[runIdx] = Cr
        Time_all[runIdx] = time() - t_run
        
        open("results_pso_run_$(runIdx).txt", "w") do f
            writedlm(f, GBestPos)
            writedlm(f, [GBestFit])
        end
    end
    
    t_total = time() - t_start
    avgTime = mean(Time_all)

    println(" ")
    println("=== RATA-RATA HASIL $Nrun RUNNING (PSO) ===")
    println("Time for exhaustive analysis (s)             : ", (round(t_analisis_total, digits=4)))
    println("Average Time per PSO run (s)                 : ", (round(avgTime, digits=4)))
    println("Rata-rata Capture Ratio (%)                  : $(round(mean(CR_all), digits=2))")
    println("Std Capture Ratio (%)                        : $(round(std(CR_all), digits=2))")
    println("Min Capture Ratio (%)                        : $(round(minimum(CR_all), digits=2))")
    println("Max Capture Ratio (%)                        : $(round(maximum(CR_all), digits=2))")
    
    println("\n=============================================================")
    println("SUCCESS: PSO completed for ", case_name)
    println("=============================================================")
end

# Execute if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    PSO_nk("case30.m", 2)
end
