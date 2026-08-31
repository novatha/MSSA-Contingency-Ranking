# MSSA_nk.jl - Optimized Modified Salp Swarm Algorithm for N-k Contingency Analysis

using PowerModels
using Ipopt
using DelimitedFiles
using Statistics
using Combinatorics
using SpecialFunctions
using Random
using Distributed
using Base.Threads

# Include the analisis_nk function
include("analisis_nk_optimized_final.jl")

"""
    generate_nk_contingencies(mpc::Dict{String, Any}, k::Int)

Generates all possible N-k contingencies for a given power system model.
"""
function generate_nk_contingencies(mpc::Dict{String, Any}, k::Int)
    if k <= 0
        error("K must be a positive integer.")
    end

    num_branches = length(mpc["branch"])

    if k > num_branches
        error("K cannot be greater than the total number of branches.")
    end

    branch_indices = 1:num_branches
    nk_contingencies = collect(combinations(branch_indices, k))

    # Convert the array of arrays to a 2D array
    return vcat(nk_contingencies'...)
end

"""
    get_contingency_probability_single(mpc::Dict{String, Any}, branch_idx::Int)

Calculates individual contingency probability based on branch reactance.
"""
function get_contingency_probability_single(mpc::Dict{String, Any}, branch_idx::Int)
    # Define a baseline failure rate (e.g., 0.01 failures/year)
    lambda_base = 0.01

    # Extract branch data
    branch_reactances = [abs(branch["br_x"]) for (i, branch) in mpc["branch"]]

    # Calculate average reactance for weighting (over all branches)
    avg_reactance = sum(branch_reactances) / length(branch_reactances)

    # Calculate probability for the specified branch contingency
    probability = lambda_base * (abs(branch_reactances[branch_idx]) / avg_reactance)

    return probability
end

"""
    repair_knapsack_solution(pos::Vector{Int}, weights::Vector{Float64}, values::Vector{Float64}, capacity::Int)

Repairs a solution to satisfy the knapsack capacity constraint by removing items with lowest value-to-weight ratio.
"""
function repair_knapsack_solution(pos::Vector{Int}, weights::Vector{Float64}, values::Vector{Float64}, capacity::Int)
    total_weight = sum(pos .* weights)
    if total_weight > capacity
        selected = findall(x -> x == 1, pos)
        if !isempty(selected)
            num_to_remove = Int(total_weight - capacity)

            if num_to_remove > 0
                # Use heuristic threshold for number of items to remove
                if num_to_remove < 5  
                    @inbounds for r_iter in 1:num_to_remove
                        current_selected_values = @view values[selected]
                        current_selected_weights = @view weights[selected]
                        current_ratios = current_selected_values ./ current_selected_weights
                        min_ratio_idx = argmin(current_ratios)
                        item_to_remove_global_idx = selected[min_ratio_idx]

                        pos[item_to_remove_global_idx] = 0
                        selected = findall(x -> x == 1, pos)
                        if isempty(selected) break end
                    end
                else  # Use sort for larger number of items to remove
                    selected_values = values[selected]
                    selected_weights = weights[selected]
                    ratios = selected_values ./ selected_weights
                    sort_idx = sortperm(ratios)
                    sorted_selected = selected[sort_idx]

                    items_to_remove_global_idx = sorted_selected[1:min(num_to_remove, length(sorted_selected))]
                    pos[items_to_remove_global_idx] .= 0
                end
            end
        end
    end
    return pos
end

"""
    evaluate_objective(position::Vector{Int}, values::Vector{Float64})

Evaluates the objective function for a single solution.
"""
function evaluate_objective(position::Vector{Int}, values::Vector{Float64})
    total = 0.0
    @inbounds @simd for i in 1:length(values)
        if position[i] == 1
            total += values[i]
        end
    end
    return total
end

"""
    update_salp_positions!(SalpPositions::Matrix{Int}, FoodPosition::Vector{Int}, c1::Float64, lb::Vector{Float64}, ub::Vector{Float64}, N::Int, dim::Int)

Updates salp positions based on the leader and follower equations.
"""
function update_salp_positions!(SalpPositions::Matrix{Int}, FoodPosition::Vector{Int}, c1::Float64, lb::Vector{Float64}, ub::Vector{Float64}, N::Int, dim::Int)
    # Leaders (first half) - updating based on food position
    N_half = Int(N/2)
    
    # Pre-generate random matrices for vectorized operations
    c2 = rand(Float64, dim, N_half)
    c3 = rand(Float64, dim, N_half)
    bin_rand1 = rand(Float64, dim, N_half)
    
    # Compute position updates for leaders using vectorized operations
    FoodPosition_float = float.(FoodPosition)  # Convert to Float64 for computation
    
    # Compute the sign matrix (2*(c3 < 0.5) - 1)
    sign_matrix = 2.0 .* (c3 .< 0.5) .- 1.0
     
    # Compute update terms
    update_terms = c1 .* ((ub .- lb) .* c2 .+ lb)
    
    # Compute temporary positions
    temp_positions = FoodPosition_float .+ update_terms .* sign_matrix
    
    # Apply sigmoid transformation and convert to binary
    X = 1.0 ./ (1.0 .+ exp.(-temp_positions))
    binary_updates = Int.(X .> bin_rand1)
    
    # Update leader positions
    @inbounds for j in 1:dim
        for i in 1:N_half
            if i <= size(SalpPositions, 1)  # Ensure we don't go out of bounds
                SalpPositions[i, j] = binary_updates[j, i]
            end
        end
    end
    
    # Followers (remaining half) - updating based on previous salp positions 
    if N > N_half
        @inbounds for i in (N_half + 1):N
            if i > 1  # Prevent index out of bounds
                # Average of position of previous salp
                prev_salp_position = SalpPositions[i-1, :]
                
                # Calculate followers' position (based on the salp before it)
                for j in 1:dim
                    SalpPositions[i, j] = Int((0.5 * prev_salp_position[j] + 0.5 * rand()) > 0.5)  # Simplified follower equation
                end
            end
        end
    end
end

"""
    mutate_salps!(SalpPositions::Matrix{Int}, mutation_rate::Float64)

Performs mutation on all salp positions.
"""
function mutate_salps!(SalpPositions::Matrix{Int}, mutation_rate::Float64)
    rows, cols = size(SalpPositions)
    
    # Vectorized mutation approach
    mutation_mask = rand(Float64, rows, cols) .< mutation_rate
    @inbounds @simd for j in 1:cols
        @simd for i in 1:rows
            if mutation_mask[i, j]
                SalpPositions[i, j] = 1 - SalpPositions[i, j]  # Flip bit
            end
        end
    end
end

"""
    MSSA_nk(case_name::String, k::Int=2)

Modified Salp Swarm Algorithm for N-k contingency analysis with performance optimizations.
"""
function MSSA_nk(case_name::String, k::Int=2; 
                 N::Int=40,        # Population size
                 Max_iter::Int=100, # Maximum iterations
                 Nrun::Int=10)     # Number of runs
    println("=== N-$k CONTINGENCY ANALYSIS for $case_name ===")
    t_analisis = time()
    mpc = PowerModels.parse_file(case_name)

    # Generate N-k contingencies
    contingencies_nk = generate_nk_contingencies(mpc, k)
    Ng = size(contingencies_nk, 1) # Total number of N-k contingencies

    println("Number of N-$k contingencies: $Ng")

    # Preallocate IP_V_SELURUH for N-k contingencies
    IP_V_SELURUH = Vector{Float64}(undef, Ng)

    # Calculate IP_V for each N-k contingency using parallel processing
    Threads.@threads for i in 1:Ng
        IP_V_SELURUH[i] = analisis_nk(mpc, contingencies_nk[i, :])
    end

    t_analisis_total = time() - t_analisis
    println("Time for N-$k analysis (s)            : $(round(t_analisis_total, digits=4))")
    println(" ")

    # Get contingency probabilities
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

    # Problem setup
    weights = ones(Ng)
    values = IP_V_SELURUH .* contingency_probabilities
    capacity = Int(round(0.5 * Ng))

    # SSA parameters
    dim = Ng
    lb = zeros(dim)
    ub = ones(dim)

    # Run parameters
    CR_all = zeros(Nrun)
    S3P_all = zeros(Nrun)
    Time_all = zeros(Nrun)

    t_start = time()
    @inbounds for runIdx in 1:Nrun
        println("Running ke-$runIdx")
        t_run = time()

        # Objective function
        evaluate(position) = evaluate_objective(position, values)

        # Initialization
        SalpPositions = rand(0:1, N, dim)
        SalpFitness = Vector{Float64}(undef, N)
        FoodPosition = zeros(Int, dim)
        FoodFitness = -Inf

        # Initial evaluation and repair
        @inbounds for i in 1:N
            pos = SalpPositions[i, :]
            SalpPositions[i, :] = repair_knapsack_solution(pos, weights, values, capacity)
            SalpFitness[i] = evaluate(SalpPositions[i, :])
            if SalpFitness[i] > FoodFitness
                FoodPosition = copy(SalpPositions[i, :])
                FoodFitness = SalpFitness[i]
            end
        end

        Convergence_curve = zeros(Max_iter)
        Convergence_curve[1] = FoodFitness
        l = 2

        # Optimal fitness for stopping criterion
        sorted_indices_for_opt_fitness = sortperm(values, rev=true)
        manual_indices_for_opt_fitness = sorted_indices_for_opt_fitness[1:min(20, length(values))]
        optimal_fitness_value = sum(values[manual_indices_for_opt_fitness])

        no_improvement_count = 0

        # Main MSSA loop with performance optimizations
        @inbounds while l <= Max_iter && FoodFitness < optimal_fitness_value && no_improvement_count < 100
            c1 = 2 * exp(-(4 * l / Max_iter)^2) # Eq. 3.2
            previous_best = copy(FoodPosition)
            previous_best_fitness = FoodFitness

            # Update salp positions
            update_salp_positions!(SalpPositions, FoodPosition, c1, lb, ub, N, dim)

            # Apply mutation
            mutation_rate = 0.05 * (1 - (l-1)/Max_iter)
            mutate_salps!(SalpPositions, mutation_rate)

            # Repair solutions to satisfy knapsack constraint
            @inbounds @simd for i in 1:N
                SalpPositions[i, :] = repair_knapsack_solution(SalpPositions[i, :], weights, values, capacity)
            end

            # Vectorized fitness evaluation for all salps
            @inbounds @simd for i in 1:N
                SalpFitness[i] = evaluate(SalpPositions[i, :])
            end

            # Find and update FoodPosition
            max_fitness, max_idx = findmax(SalpFitness)
            if max_fitness > FoodFitness
                FoodPosition = copy(SalpPositions[max_idx, :])
                FoodFitness = max_fitness
                no_improvement_count = 0 # Reset counter if improvement
            else
                no_improvement_count += 1 # Increment if no improvement
            end

            # Elitism - preserve best solution
            min_fitness_current, min_idx = findmin(SalpFitness)
            if previous_best_fitness > min_fitness_current
                SalpPositions[min_idx, :] = previous_best
                SalpFitness[min_idx] = previous_best_fitness

                if previous_best_fitness > FoodFitness
                    FoodPosition = previous_best
                    FoodFitness = previous_best_fitness
                end
            end

            Convergence_curve[l] = FoodFitness
            l += 1
        end

        # --- Local Search (Hill Climbing) on FoodPosition ---
        current_best_solution_ls = copy(FoodPosition) # Start local search from MSSA's best
        current_best_fitness_ls = FoodFitness

        improved_ls = true
        while improved_ls
            improved_ls = false

            @inbounds @simd for j in 1:dim # Iterate through each contingency
                # Try flipping the bit
                test_pos_ls = copy(current_best_solution_ls)
                test_pos_ls[j] = 1 - test_pos_ls[j]

                # Repair if over capacity
                test_pos_ls = repair_knapsack_solution(test_pos_ls, weights, values, capacity)
                test_fitness_ls = evaluate(test_pos_ls)

                # If improvement found, update and continue
                if test_fitness_ls > current_best_fitness_ls
                    current_best_fitness_ls = test_fitness_ls
                    current_best_solution_ls = test_pos_ls
                    improved_ls = true
                end
            end
        end

        # Update FoodPosition with the locally optimized solution
        FoodPosition = current_best_solution_ls
        FoodFitness = current_best_fitness_ls

        # --- Results ---
        manual_idx = sortperm(values, rev=true)
        manual_indices_top20 = manual_idx[1:min(20, length(values))]

        selected_idx = findall(x -> x == 1, FoodPosition)
        sort_order = sortperm(values[selected_idx] .* weights[selected_idx], rev=true)
        ssa_indices_top20 = selected_idx[sort_order[1:min(20, length(sort_order))]]

        persamaan = in.(manual_indices_top20, (Set(ssa_indices_top20),))
        Ncr = sum(persamaan)
        Nm = length(manual_indices_top20)
        Cr = (Ncr/Nm) * 100

        q1a1 = N * Max_iter
        kombinasi_log = lgamma(Ng+1) - lgamma(Nm+1) - lgamma(Ng-Nm+1)
        kombinasi = exp(kombinasi_log)
        S3P = (q1a1 / kombinasi) * 100

        CR_all[runIdx] = Cr
        S3P_all[runIdx] = S3P
        Time_all[runIdx] = time() - t_run
        
        # Save convergence curve
        open("convergence_mssa_run_$(runIdx).txt", "w") do f
            writedlm(f, Convergence_curve[1:l-1])  # Only save up to the actual number of iterations
        end
        
        open("results_mssa_run_$(runIdx).txt", "w") do f
            writedlm(f, FoodPosition)
            writedlm(f, [FoodFitness])
        end
    end
    t_total = time() - t_start
    avgTime = mean(Time_all)

    # --- Display summary ---
    println(" ")
    println("=== RATA-RATA HASIL $Nrun RUNNING ===")
    println("Time for analisis_all (base case) (s)        : $(round(t_analisis_total, digits=4))")
    println("Average Time per MSSA run (s)                : $(round(avgTime, digits=4))")
    println("Total Time (analisis + 10 runs) (s)          : $(round(t_analisis_total + t_total, digits=4))")
    println("Rata-rata Capture Ratio (%)                   : $(round(mean(CR_all), digits=2))")
    println("Rata-rata Search Space Scan Percentage (%)    : $(mean(S3P_all))")
    println("=== STANDAR DEVIASI ===")
    println("Std Time per run (s)                        : $(round(std(Time_all), digits=4))")
    println("Std Capture Ratio (%)                       : $(round(std(CR_all), digits=2))")
    println("Std Search Space Scan Percentage (%)        : $(std(S3P_all))")
    println("=== MAX DAN MIN ===")
    println("Max Time per run (s)                        : $(round(maximum(Time_all), digits=4))")
    println("Min Time per run (s)                        : $(round(minimum(Time_all), digits=4))")
    println("Max Capture Ratio (%)                       : $(round(maximum(CR_all), digits=2))")
    println("Min Capture Ratio (%)                       : $(round(minimum(CR_all), digits=2))")
    println("Max Search Space Scan Percentage (%)        : $(maximum(S3P_all))")
    println("Min Search Space Scan Percentage (%)        : $(minimum(S3P_all))")
    
    println("\\n=============================================================")
    println("SUCCESS: MSSA completed for $case_name")
    println("Results have been saved to output files.")
    println("=============================================================")
end

# Example usage:
# MSSA_nk("case57.m", 2)