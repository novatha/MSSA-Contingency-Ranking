# MSSA_direct_nk.jl - Direct On-The-Fly N-k Contingency Search using Modified Salp Swarm Algorithm
# This version searches directly in the space of branch outages without requiring exhaustive upfront evaluation.

using PowerModels
using Ipopt
using Random
using Statistics
using Printf
using Base.Threads

include("contingency_utils.jl")
include("analisis_nk_optimized_final.jl")

"""
    ArchiveEntry
Structure to maintain unique top-K contingencies found during the search.
"""
struct ArchiveEntry
    contingency::Vector{Int}
    risk_index::Float64
    ip_v::Float64
    probability::Float64
end

"""
    DirectMSSAConfig
Configuration parameters for direct N-k search.
"""
struct DirectMSSAConfig
    N::Int              # Swarm population size (e.g. 30)
    Max_iter::Int       # Maximum iterations (e.g. 50)
    top_k::Int          # Number of top critical contingencies to identify (e.g. 20)
    mutation_rate::Float64 # Base mutation rate
end

function DirectMSSAConfig(; N=30, Max_iter=50, top_k=20, mutation_rate=0.15)
    return DirectMSSAConfig(N, Max_iter, top_k, mutation_rate)
end

"""
    repair_contingency(cont::Vector{Int}, num_branches::Int)
Ensures the k branch indices are valid (1 to num_branches), strictly unique, and sorted.
"""
function repair_contingency(cont::Vector{Int}, num_branches::Int, k::Int)
    clamped = [clamp(b, 1, num_branches) for b in cont]
    unique_branches = unique(clamped)
    
    # If duplicates occurred, randomly sample missing slots
    while length(unique_branches) < k
        candidate = rand(1:num_branches)
        if !(candidate in unique_branches)
            push!(unique_branches, candidate)
        end
    end
    
    return sort(unique_branches[1:k])
end

"""
    evaluate_contingency_direct(mpc, contingency, cache, eval_counter)
Evaluates an N-k contingency on-the-fly with caching.
"""
function evaluate_contingency_direct(
    mpc::Dict{String, Any}, 
    contingency::Vector{Int}, 
    cache::Dict{Vector{Int}, Tuple{Float64, Float64, Float64}},
    eval_counter::Ref{Int}
)
    if haskey(cache, contingency)
        return cache[contingency]
    end

    # Evaluate AC power flow on the fly
    eval_counter[] += 1
    ip_v = analisis_nk(mpc, contingency)

    # Compute joint probability
    joint_prob = 1.0
    for b in contingency
        joint_prob *= get_contingency_probability_single(mpc, b)
    end

    # Handle non-convergent / severe cases
    risk_index = ip_v >= 1e9 ? 1e9 * joint_prob : ip_v * joint_prob

    result = (risk_index, ip_v, joint_prob)
    cache[contingency] = result
    return result
end

"""
    update_archive!(archive, entry, max_size)
Maintains the global Top-K unique contingencies.
"""
function update_archive!(archive::Vector{ArchiveEntry}, entry::ArchiveEntry, max_size::Int)
    # Check if already in archive
    for item in archive
        if item.contingency == entry.contingency
            return
        end
    end

    push!(archive, entry)
    sort!(archive, by = x -> x.risk_index, rev = true)
    if length(archive) > max_size
        pop!(archive)
    end
end

"""
    run_direct_mssa(case_name::String, k::Int; config=DirectMSSAConfig())
Executes Direct On-The-Fly MSSA for N-k contingency ranking.
"""
function run_direct_mssa(case_name::String, k::Int=2; config=DirectMSSAConfig())
    println("=================================================================")
    println(" DIRECT ON-THE-FLY MSSA N-$k CONTINGENCY RANKING")
    println(" System: $case_name | k = $k (Simultaneous Outages)")
    println("=================================================================")

    t_start = time()

    # Resolve file path
    actual_case = case_name
    if !isfile(actual_case)
        if isfile(actual_case * ".m")
            actual_case = actual_case * ".m"
        elseif isfile(joinpath("test_cases", actual_case))
            actual_case = joinpath("test_cases", actual_case)
        elseif isfile(joinpath("test_cases", actual_case * ".m"))
            actual_case = joinpath("test_cases", actual_case * ".m")
        elseif isfile(joinpath(@__DIR__, "test_cases", actual_case))
            actual_case = joinpath(@__DIR__, "test_cases", actual_case)
        elseif isfile(joinpath(@__DIR__, "test_cases", actual_case * ".m"))
            actual_case = joinpath(@__DIR__, "test_cases", actual_case * ".m")
        end
    end
    mpc = PowerModels.parse_file(actual_case)
    num_branches = length(mpc["branch"])
    
    total_combinations = binomial(num_branches, k)
    println("Total Search Space: $(total_combinations) possible N-$k combinations")
    println("Population size (N): $(config.N), Iterations: $(config.Max_iter)")
    
    cache = Dict{Vector{Int}, Tuple{Float64, Float64, Float64}}()
    eval_counter = Ref(0)
    archive = Vector{ArchiveEntry}()

    # --- Phase 1: Population Initialization ---
    dim = k
    lb = 1.0
    ub = Float64(num_branches)

    # Continuous positions
    SalpPositions = rand(config.N, dim) .* (ub - lb) .+ lb
    SalpFitness = zeros(Float64, config.N)
    
    FoodPosition = zeros(Float64, dim)
    FoodContingency = zeros(Int, dim)
    FoodFitness = -Inf

    for i in 1:config.N
        discrete_cont = repair_contingency(round.(Int, SalpPositions[i, :]), num_branches, k)
        SalpPositions[i, :] .= Float64.(discrete_cont)
        
        ri, ip_v, prob = evaluate_contingency_direct(mpc, discrete_cont, cache, eval_counter)
        SalpFitness[i] = ri
        update_archive!(archive, ArchiveEntry(discrete_cont, ri, ip_v, prob), config.top_k)

        if ri > FoodFitness
            FoodFitness = ri
            FoodPosition .= SalpPositions[i, :]
            FoodContingency .= discrete_cont
        end
    end

    # --- Phase 2: Main Swarm Optimization Loop ---
    for l in 1:config.Max_iter
        c1 = 2.0 * exp(-(4.0 * l / config.Max_iter)^2)
        p_m = config.mutation_rate * (1.0 - (l - 1) / config.Max_iter)

        # Leaders Update (First half)
        N_half = div(config.N, 2)
        for i in 1:N_half
            for j in 1:dim
                c2 = rand()
                c3 = rand()
                sign_val = (c3 < 0.5) ? -1.0 : 1.0
                SalpPositions[i, j] = FoodPosition[j] + c1 * ((ub - lb) * c2 + lb) * sign_val
            end
        end

        # Followers Update (Second half)
        for i in (N_half + 1):config.N
            for j in 1:dim
                SalpPositions[i, j] = 0.5 * (SalpPositions[i, j] + SalpPositions[i - 1, j])
            end
        end

        # Adaptive Mutation & Evaluation
        for i in 1:config.N
            # Random mutation of branch index
            if rand() < p_m
                rand_dim = rand(1:dim)
                SalpPositions[i, rand_dim] = rand() * (ub - lb) + lb
            end

            discrete_cont = repair_contingency(round.(Int, SalpPositions[i, :]), num_branches, k)
            SalpPositions[i, :] .= Float64.(discrete_cont)

            ri, ip_v, prob = evaluate_contingency_direct(mpc, discrete_cont, cache, eval_counter)
            SalpFitness[i] = ri
            update_archive!(archive, ArchiveEntry(discrete_cont, ri, ip_v, prob), config.top_k)

            if ri > FoodFitness
                FoodFitness = ri
                FoodPosition .= SalpPositions[i, :]
                FoodContingency .= discrete_cont
            end
        end

        # Local Neighborhood Exploration around Best Leader
        for j in 1:dim
            for delta in [-2, -1, 1, 2]
                neighbor_cont = copy(FoodContingency)
                neighbor_cont[j] = clamp(neighbor_cont[j] + delta, 1, num_branches)
                neighbor_cont = repair_contingency(neighbor_cont, num_branches, k)
                
                ri, ip_v, prob = evaluate_contingency_direct(mpc, neighbor_cont, cache, eval_counter)
                update_archive!(archive, ArchiveEntry(neighbor_cont, ri, ip_v, prob), config.top_k)
                if ri > FoodFitness
                    FoodFitness = ri
                    FoodContingency .= neighbor_cont
                    FoodPosition .= Float64.(neighbor_cont)
                end
            end
        end
    end

    t_elapsed = time() - t_start

    # --- Phase 3: Results Display ---
    println("\n==================== SEARCH COMPLETED ====================")
    println(@sprintf("Execution Time:                %.4f seconds", t_elapsed))
    println(@sprintf("Unique Contingencies Evaluated: %d / %d (%.4f%% of search space)", 
        eval_counter[], total_combinations, (eval_counter[] / total_combinations) * 100))
    println(@sprintf("Speedup vs Exhaustive Search:  ~%.1fx faster", (total_combinations / max(1, eval_counter[]))))
    println("==========================================================")

    println("\nTop Identified Critical Contingencies (Ranked by Risk Index):")
    println("Rank | Outaged Branches | Risk Index  | Voltage Severity (IP_V) | Joint Probability")
    println("-----------------------------------------------------------------------------------")
    for (idx, entry) in enumerate(archive)
        println(@sprintf(" %2d  | %-16s | %10.4e | %22.4f | %17.4e", 
            idx, string(entry.contingency), entry.risk_index, entry.ip_v, entry.probability))
    end
    println("-----------------------------------------------------------------------------------")

    return archive, t_elapsed, eval_counter[]
end

# Main entry point
if abspath(PROGRAM_FILE) == @__FILE__
    case = length(ARGS) >= 1 ? ARGS[1] : "casesumatera.m"
    k_val = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
    run_direct_mssa(case, k_val)
end
