# Julia translation of GA_nk.m with maximum performance optimizations

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
    const lambda_base = 0.01

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
    total_weight = sum(@view pos .* weights)
    if total_weight > capacity
        selected = findall(x -> x == 1, pos)
        if !isempty(selected)
            num_to_remove = Int(total_weight - capacity)

            if num_to_remove > 0
                if num_to_remove < 5 # Heuristic threshold for iterative min
                    @inbounds for r_iter in 1:num_to_remove
                        current_selected_values = values[selected]
                        current_selected_weights = weights[selected]
                        current_ratios = current_selected_values ./ current_selected_weights
                        min_ratio_idx = argmin(current_ratios)
                        item_to_remove_global_idx = selected[min_ratio_idx]

                        pos[item_to_remove_global_idx] = 0
                        selected = findall(x -> x == 1, pos)
                        if isempty(selected) break end
                    end
                else # Use sort for larger number of items to remove
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
    evaluate_fitness(population::Matrix{Int}, values::Vector{Float64})

Vectorized fitness evaluation for entire population.
"""
function evaluate_fitness(population::Matrix{Int}, values::Vector{Float64})
    n_rows = size(population, 1)
    result = Vector{Float64}(undef, n_rows)
    
    @threads for i in 1:n_rows
        row_sum = 0.0
        @inbounds @simd for j in 1:length(values)
            if population[i, j] == 1
                row_sum += values[j]
            end
        end
        result[i] = row_sum
    end
    return result
end

"""
    select_parents_roulette(fitness::Vector{Float64}, num_pairs::Int)

Selects parent pairs using roulette wheel selection.
"""
function select_parents_roulette(fitness::Vector{Float64}, num_pairs::Int)
    totalFit = sum(fitness)
    if totalFit == 0
        probs = fill(1.0 / length(fitness), length(fitness))
    else
        probs = fitness / totalFit
    end
    cumProb = cumsum(probs)
    
    # Pre-generate random numbers
    rand_values = rand(2 * num_pairs)
    
    p1_indices = Vector{Int}(undef, num_pairs)
    p2_indices = Vector{Int}(undef, num_pairs)
    
    @inbounds for i in 1:num_pairs
        p1_indices[i] = findfirst(x -> x >= rand_values[2*i-1], cumProb)
        p2_indices[i] = findfirst(x -> x >= rand_values[2*i], cumProb)
        if p1_indices[i] === nothing
            p1_indices[i] = length(fitness)
        end
        if p2_indices[i] === nothing
            p2_indices[i] = length(fitness)
        end
    end
    
    return p1_indices, p2_indices
end

"""
    crossover_operator(parent1::Vector{Int}, parent2::Vector{Int}, pcross::Float64)

Performs crossover between two parent solutions.
"""
function crossover_operator(parent1::Vector{Int}, parent2::Vector{Int}, pcross::Float64)
    if rand() < pcross
        pt = rand(1:length(parent1)-1)
        child1 = vcat(view(parent1, 1:pt), view(parent2, pt+1:lastindex(parent2)))
        child2 = vcat(view(parent2, 1:pt), view(parent1, pt+1:lastindex(parent1)))
    else
        child1 = copy(parent1)
        child2 = copy(parent2)
    end
    return child1, child2
end

"""
    mutate_solution!(solution::Vector{Int}, pmut::Float64)

Performs mutation on a solution.
"""
function mutate_solution!(solution::Vector{Int}, pmut::Float64)
    n = length(solution)
    # Vectorized approach to mutation
    mutation_mask = rand(n) .< pmut
    @inbounds @simd for i in 1:n
        if mutation_mask[i]
            solution[i] = 1 - solution[i]
        end
    end
    return solution
end

"""
    GA_nk(case_name::String, k::Int=2)

Genetic Algorithm for N-k contingency analysis with maximum performance optimizations.
"""
function GA_nk(case_name::String, k::Int=2; 
               popSize::Int=50, 
               numGen::Int=200,
               pcross::Float64=0.9,
               elitism::Bool=true,
               Nrun::Int=10)
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
    values = IP_V_SELURUH .* contingency_probabilities
    weights = ones(Ng)
    capacity = Int(round(0.5 * Ng))

    # Run parameters
    CR_all = zeros(Nrun)
    S3P_all = zeros(Nrun)
    Time_all = zeros(Nrun)

    t_start = time()
    @inbounds for runIdx in 1:Nrun
        println("Running ke-$runIdx")
        t_run = time()

        # Initialization
        population = rand(0:1, popSize, Ng)
        @inbounds for i in 1:popSize
            population[i, :] = repair_knapsack_solution(population[i, :], weights, values, capacity)
        end

        bestFitness = -Inf
        bestChrom = zeros(Int, Ng)  # Changed to Int type
        fitnessHistory = zeros(numGen)
        no_improvement_count = 0

        # Main GA loop
        @inbounds for gen in 1:numGen
            pmut = 0.05 * (1 - (gen / numGen))

            # Vectorized fitness evaluation
            fitness = evaluate_fitness(population, values)

            # Store best solution
            fmax, idxMax = findmax(fitness)
            fitnessHistory[gen] = fmax
            if fmax > bestFitness
                bestFitness = fmax
                bestChrom = copy(population[idxMax[1], :])
                no_improvement_count = 0
            else
                no_improvement_count += 1
            end

            if no_improvement_count >= 100
                break
            end

            # Selection (Roulette Wheel)
            p1_indices, p2_indices = select_parents_roulette(fitness, Int(ceil(popSize / 2)))

            # Generate new population
            newPop = similar(population)

            # Elitism
            if elitism
                newPop[1, :] = bestChrom
                startIdx = 2
            else
                startIdx = 1
            end

            # Create offspring pairs
            pair_counter = 0
            @inbounds for k_ga in startIdx:2:popSize
                pair_counter += 1
                
                p1 = population[p1_indices[pair_counter], :]
                p2 = population[p2_indices[pair_counter], :]

                c1, c2 = crossover_operator(p1, p2, pcross)

                # Apply mutation and repair
                # Child 1
                c1 = mutate_solution!(copy(c1), pmut)
                c1 = repair_knapsack_solution(c1, weights, values, capacity)
                newPop[k_ga, :] = c1

                # Child 2
                if (k_ga + 1) <= popSize
                    c2 = mutate_solution!(copy(c2), pmut)
                    c2 = repair_knapsack_solution(c2, weights, values, capacity)
                    newPop[k_ga + 1, :] = c2
                end
            end
            
            population = newPop
        end

        # --- Results ---
        manual_idx = sortperm(values, rev=true)
        manual_indices_top20 = manual_idx[1:min(20, length(values))]

        selected_idx = findall(x -> x == 1, bestChrom)
        sorted_indices = sortperm(values[selected_idx] .* weights[selected_idx], rev=true)
        
        top_ga_indices = Int[]
        for i in 1:min(20, length(sorted_indices))
            push!(top_ga_indices, selected_idx[sorted_indices[i]])
        end

        persamaan = in.(manual_indices_top20, (Set(top_ga_indices),))
        Ncr = sum(persamaan)
        Nm = length(manual_indices_top20)
        Cr = (Ncr/Nm) * 100

        q1a1 = popSize * numGen
        kombinasi_log = lgamma(Ng + 1) - lgamma(Nm + 1) - lgamma(Ng - Nm + 1)
        kombinasi = exp(kombinasi_log)
        S3P = (q1a1 / kombinasi) * 100

        CR_all[runIdx] = Cr
        S3P_all[runIdx] = S3P
        Time_all[runIdx] = time() - t_run
        
        # Save convergence curve
        open("convergence_ga_nk_run_$(runIdx).txt", "w") do f
            writedlm(f, fitnessHistory[1:gen])  # Only save up to the actual number of generations
        end
    end
    t_total = time() - t_start
    avgTime = mean(Time_all)

    # --- Display summary ---
    println(" ")
    println("=== RATA-RATA HASIL $Nrun RUNNING ===")
    println("Time for analisis_all (base case) (s)        : $(round(t_analisis_total, digits=4))")
    println("Average Time per GA run (s)                : $(round(avgTime, digits=4))")
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
end

# Example usage:
# GA_nk("case57.m", 2)