# Power System Contingency Analysis Utilities
# Common functions used across all analysis modules

using PowerModels
using Ipopt
using Combinatorics

# Suppress solver output
const PM_SOLVER = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

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
                if num_to_remove < 5 # Heuristic threshold for iterative min
                    for r_iter in 1:num_to_remove
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
    calculate_capture_ratio(top_manual_indices::Vector{Int}, top_algorithm_indices::Vector{Int})

Calculates the capture ratio between manual and algorithm-selected indices.
"""
function calculate_capture_ratio(top_manual_indices::Vector{Int}, top_algorithm_indices::Vector{Int})
    Nm = length(top_manual_indices)
    Ncr = sum(in.(top_manual_indices, Ref(Set(top_algorithm_indices))))
    Cr = (Ncr / Nm) * 100
    return Cr
end

"""
    calculate_search_space_percentage(N::Int, Max_iter::Int, Ng::Int, Nm::Int)

Calculates the search space scan percentage.
"""
function calculate_search_space_percentage(N::Int, Max_iter::Int, Ng::Int, Nm::Int)
    q1a1 = N * Max_iter
    kombinasi_log = lgamma(Ng + 1) - lgamma(Nm + 1) - lgamma(Ng - Nm + 1)
    kombinasi = exp(kombinasi_log)
    S3P = (q1a1 / kombinasi) * 100
    return S3P
end