# Julia translation of analisis_nk.m with maximum performance optimizations

using PowerModels
using Ipopt # Or another solver
using Graphs # For connected_components

# Suppress solver output
const PM_SOLVER = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0) # Set back to 0 for normal operation

"""
    analisis_nk(mpc::Dict{String, Any}, contingency_indices::Vector{Int})

Performs N-k contingency analysis on a power system model.

# Arguments
- `mpc::Dict{String, Any}`: Power system data in PowerModels format
- `contingency_indices::Vector{Int}`: Vector of branch indices to be outaged

# Returns
- Float64: Voltage performance index (1e10 if contingency fails)
"""
function analisis_nk(mpc::Dict{String, Any}, contingency_indices::Vector{Int})
    # Validate inputs
    @inbounds for idx in contingency_indices
        if !haskey(mpc["branch"], string(idx))
            @warn "Branch $idx not found in network.branch"
            return 1e10 # Mark as failed if branch not found
        end
    end

    # Create a copy of the network data to modify - only copy what's needed
    network_contingency = copy_network_contingency(mpc, contingency_indices)

    # Define penalty parameters
    vmin = 0.95       # p.u.
    vmax = 1.05       # p.u.
    Vsp = 1.00        # p.u.
    deltaV_lim = (vmax - vmin) / 2
    z = 1.0

    total_penalty::Float64 = 0.0

    # Handle islanding using PowerModels.jl's connected components
    components = PowerModels.calc_connected_components(network_contingency)

    # Process each island
    @inbounds for comp_buses in components
        # Convert to Vector{Int} if needed (components may return different types)
        comp_buses_vec = collect(Int, comp_buses)
        island_data = create_island_data(network_contingency, comp_buses_vec)
        
        # Check if island is viable (has generators and proper slack bus assignment)
        if !is_island_viable!(island_data)
            return 1e10  # Contingency failed
        end

        # Run power flow for the island
        result = PowerModels.solve_ac_pf(island_data, PM_SOLVER)
        
        if result["termination_status"] != LOCALLY_SOLVED
            return 1e10  # Power flow failed
        end

        # Calculate voltage penalty for this island
        total_penalty += calculate_voltage_penalty(result["solution"]["bus"], vmin, vmax, Vsp, deltaV_lim, z)
    end

    return total_penalty
end

"""
    copy_network_contingency(mpc::Dict{String, Any}, contingency_indices::Vector{Int})

Efficiently creates a copy of the network data with specified branches outaged.
"""
function copy_network_contingency(mpc::Dict{String, Any}, contingency_indices::Vector{Int})
    network_contingency = Dict{String, Any}()
    
    # Copy top-level keys
    for key in keys(mpc)
        if key != "branch"  # Handle branch separately to avoid deep copy of all branches
            @inbounds network_contingency[key] = mpc[key]
        end
    end
    
    # Copy branch data efficiently
    network_contingency["branch"] = Dict{String, Any}()
    @inbounds for (br_id, br_data) in mpc["branch"]
        network_contingency["branch"][br_id] = br_data  # Avoid deep copy
    end
    
    # Set the status of the branches in the contingency to out of service (0)
    @inbounds for idx in contingency_indices
        branch_id = string(idx)
        if haskey(network_contingency["branch"], branch_id)
            network_contingency["branch"][branch_id]["br_status"] = 0
        end
    end
    
    return network_contingency
end

"""
    create_island_data(network_contingency::Dict{String, Any}, comp_buses::Vector{Int})

Creates a sub-network data dictionary for a connected component (island).
"""
function create_island_data(network_contingency::Dict{String, Any}, comp_buses::Vector{Int})
    # Initialize island data structure
    island_data = Dict{String, Any}()
    island_data["bus"] = Dict{String, Any}()
    island_data["branch"] = Dict{String, Any}()
    island_data["gen"] = Dict{String, Any}()
    island_data["load"] = Dict{String, Any}()
    island_data["shunt"] = Dict{String, Any}()
    island_data["dcline"] = Dict{String, Any}()
    island_data["storage"] = Dict{String, Any}()
    island_data["switch"] = Dict{String, Any}()

    # Copy top-level keys that are present in network_contingency
    for key in ["baseMVA", "per_unit", "name", "source_type", "source_version"]
        if haskey(network_contingency, key)
            @inbounds island_data[key] = network_contingency[key]
        end
    end

    # Create bus mapping (old bus ID to new bus ID)
    bus_map = Dict{Int, Int}()
    new_bus_idx = 1
    
    # Pre-compute bus mappings
    @inbounds @simd for bus_id in comp_buses
        old_bus_data = network_contingency["bus"][string(bus_id)]
        new_bus_data = copy(old_bus_data)  # Shallow copy
        new_bus_data["bus_i"] = new_bus_idx
        new_bus_data["index"] = new_bus_idx
        island_data["bus"][string(new_bus_idx)] = new_bus_data
        bus_map[bus_id] = new_bus_idx
        new_bus_idx += 1
    end

    # Copy branches that connect buses in this island
    @inbounds for (br_id, br_data) in network_contingency["branch"]
        # Check if both ends of the branch are in the component
        if br_data["br_status"] == 1 && 
           haskey(bus_map, br_data["f_bus"]) && 
           haskey(bus_map, br_data["t_bus"])
            new_br_data = copy(br_data)  # Shallow copy
            new_br_data["f_bus"] = bus_map[br_data["f_bus"]]
            new_br_data["t_bus"] = bus_map[br_data["t_bus"]]
            island_data["branch"][br_id] = new_br_data
        end
    end

    # Copy generators connected to buses in this island
    @inbounds for (gen_id, gen_data) in network_contingency["gen"]
        if haskey(bus_map, gen_data["gen_bus"])
            new_gen_data = copy(gen_data)  # Shallow copy
            new_gen_data["gen_bus"] = bus_map[gen_data["gen_bus"]]
            new_gen_data["index"] = parse(Int, gen_id)
            island_data["gen"][gen_id] = new_gen_data
        end
    end

    # Copy loads connected to buses in this island
    @inbounds for (load_id, load_data) in network_contingency["load"]
        if haskey(bus_map, load_data["load_bus"])
            new_load_data = copy(load_data)  # Shallow copy
            new_load_data["load_bus"] = bus_map[load_data["load_bus"]]
            island_data["load"][load_id] = new_load_data
        end
    end

    # Copy shunts connected to buses in this island
    @inbounds for (shunt_id, shunt_data) in network_contingency["shunt"]
        if haskey(bus_map, shunt_data["shunt_bus"])
            new_shunt_data = copy(shunt_data)  # Shallow copy
            new_shunt_data["shunt_bus"] = bus_map[shunt_data["shunt_bus"]]
            island_data["shunt"][shunt_id] = new_shunt_data
        end
    end

    return island_data
end

"""
    is_island_viable!(island_data::Dict{String, Any})

Checks if an island has viable generators and properly assigns a slack bus if needed.
Modifies the island_data in place to assign a slack bus if needed.
"""
function is_island_viable!(island_data::Dict{String, Any})
    # Check if the island has any online generators
    if !haskey(island_data, "gen") || isempty(island_data["gen"])
        return false
    end

    # Pre-allocate arrays to reduce allocations
    online_gen_buses = Int[]
    online_gens = []
    
    # Find online generators more efficiently
    @inbounds for (g, gen) in island_data["gen"]
        if gen["gen_status"] > 0
            push!(online_gens, gen)
            push!(online_gen_buses, gen["gen_bus"])
        end
    end
    
    if isempty(online_gens)
        return false
    end

    # Check for a functional slack bus
    ref_buses = Int[]
    @inbounds for (b, bus) in island_data["bus"]
        if bus["bus_type"] == 3
            push!(ref_buses, parse(Int, b))
        end
    end

    if isempty(intersect(online_gen_buses, ref_buses))
        # If no slack bus, assign the largest generator as the slack bus
        if !isempty(online_gens)
            # Find generator with maximum pmax efficiently
            max_pmax = -Inf
            max_pmax_gen = nothing
            @inbounds for gen in online_gens
                if gen["pmax"] > max_pmax
                    max_pmax = gen["pmax"]
                    max_pmax_gen = gen
                end
            end
            
            if max_pmax_gen !== nothing
                slack_bus_idx = max_pmax_gen["gen_bus"]

                # Find the corresponding bus and set it as slack
                @inbounds for (bus_key, bus_data) in island_data["bus"]
                    if bus_data["bus_i"] == slack_bus_idx
                        island_data["bus"][bus_key]["bus_type"] = 3
                        return true
                    else
                        if bus_data["bus_type"] == 3
                            island_data["bus"][bus_key]["bus_type"] = 1 # Set to PQ
                        end
                    end
                end
                return false
            else
                return false
            end
        else
            return false
        end
    end

    return true
end

"""
    calculate_voltage_penalty(bus_solution::Dict{String, Any}, vmin::Float64, vmax::Float64, Vsp::Float64, deltaV_lim::Float64, z::Float64)

Calculates the voltage penalty based on bus voltage magnitudes.
"""
function calculate_voltage_penalty(bus_solution::Dict{String, Any}, vmin::Float64, vmax::Float64, Vsp::Float64, deltaV_lim::Float64, z::Float64)
    total_penalty = 0.0

    @inbounds for (b, bus) in bus_solution
        vm = bus["vm"]
        if vm < vmin
            penalty = ((vm - Vsp) / deltaV_lim)^(2 * z)
            total_penalty += penalty
        elseif vm > vmax
            penalty = ((vm - Vsp) / deltaV_lim)^(2 * z)
            total_penalty += penalty
        end
    end

    return total_penalty
end

# Example of how to use the function (for testing)
# if abspath(PROGRAM_FILE) == @__FILE__
#     # Load a MATPOWER case file
#     mpc_data = PowerModels.parse_file("case57.m")
#
#     # Define a contingency (e.g., outage of branches 1 and 2)
#     contingency = [1, 2]
#
#     # Run the analysis
#     ip_v = analisis_nk(mpc_data, contingency)
#
#     println("Voltage Performance Index (IP_V): ", ip_v)
# end