"""
Kruskal computes a minimum spanning tree using Kruskal's algorithm.

Parameters:
- `instance::Instance`: The problem instance

Returns:
- `uf::UnionFind`: A UnionFind object representing the minimum spanning tree
"""
function run_kruskal(instance::Instance)
    g = instance.graph
    G = deepcopy(g)
    nb = instance.num_blocks
    g_edges =
        [Edge(src(i), dst(i), get_prop(G, i, :weight)) for i in edges(G) if src(i) != nb +1 &&
                                                                                         dst(
            i,
        ) != nb +1]
    sorted_edges = sort(g_edges, by = x -> x.weight)
    uf = UnionFind.UnionFinder(nb)
    for e in sorted_edges
        u, v, w = e.src, e.dst, e.weight
        if (find!(uf, u) != find!(uf, v)) &&
           (size!(uf, u) + size!(uf, v) <= instance.target_district_size)
            UnionFind.union!(uf, u, v)
        end
    end

    return uf
end


"""
findNeighborDistricts finds the neighboring districts of each district.

Parameters:
- `instance::Instance`: The problem instance
- `districts::Vector{Vector{Int}}`: A vector of vectors of nodes representing the districts

Returns:
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping a pair of neighboring districts to the number of nodes in the pair
"""
function findNeighborDistricts(instance::Instance, districts::Vector{Vector{Int}})
    neighborPairs = Dict{Tuple{Int,Int},Int}()
    g = instance.graph
    for i = 1:length(districts)
        for u in districts[i]
            for v in neighbors(g, u)
                if (v == instance.num_blocks + 1)
                    continue
                end
                found_v = false
                for j = (i+1):length(districts)
                    if v in districts[j]
                        node_sum = length(districts[i]) + length(districts[j])
                        neighborPairs[(i, j)] = get(neighborPairs, (i, j), node_sum)
                        found_v = true
                        break
                    end
                end
                if found_v
                    break
                end
            end
        end
    end

    return neighborPairs
end


"""
mergeDistricts merges two districts.

Parameters:
- `uf::UnionFinder`: A UnionFind object representing the minimum spanning tree
- `d1::Vector{Int}`: A vector of nodes representing the first district
- `d2::Vector{Int}`: A vector of nodes representing the second district
"""
function mergeDistricts(uf::UnionFinder, d1::Vector{Int}, d2::Vector{Int})
    for u in d1
        for v in d2
            UnionFind.union!(uf, u, v)
        end
    end
end


"""
groupBlocks groups the nodes in the minimum spanning tree into districts.

Parameters:
- `uf::UnionFind`: A UnionFind object representing the minimum spanning tree

Returns:
- `districts::Vector{Vector{Int}}`: A vector of vectors of nodes representing the districts
"""
function groupBlocks(uf::UnionFinder)
    cf = CompressedFinder(uf)
    groups = cf.groups
    district = Vector{Vector{Int}}(undef, groups)
    for i = 1:groups
        district[i] = Int[]
    end
    for i = 1:length(uf)
        k = find(cf, i)
        push!(district[k], i)
    end
    return district
end


"""
GreedyMerging merges districts greedily to repair the solution.

Parameters:
- `instance::Instance`: The problem instance
- `uf::UnionFind`: A UnionFind object representing the minimum spanning tree

Returns:
- `districts::Vector{Vector{Int}}`: A vector of vectors of nodes representing the districts
"""
function GreedyMerging(instance::Instance, uf::UnionFinder)
    targetNbDistricts = Int(floor(instance.num_blocks / instance.target_district_size))
    cf = CompressedFinder(uf)

    while cf.groups > targetNbDistricts
        districts = groupBlocks(uf)
        neighborPairs = findNeighborDistricts(instance, districts)
        minWeight = Inf
        mergePair = (0, 0)
        for (pair, weight) in neighborPairs
            if weight < minWeight
                minWeight = weight
                mergePair = pair
            end
        end

        mergeDistricts(uf, districts[mergePair[1]], districts[mergePair[2]])
        cf = CompressedFinder(uf)
    end

    districts = groupBlocks(uf)
    return [Vector{Int64}(subgraph) for subgraph in districts]
end

"""
    find_id_neighbors(i::Int, neighborPairs::Dict{Tuple{Int, Int}, Int})

Finds the IDs of neighboring districts.

# Arguments
- `i::Int`: The district ID.
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping pairs of neighboring districts to their ID.

# Returns
- A list of IDs for the neighboring districts.
"""

function find_id_neighbors(i::Int, neighborPairs::Dict{Tuple{Int,Int},Int})
    neighborIds = []
    for (pair, _) in neighborPairs
        if pair[1] == i
            push!(neighborIds, pair[2])
        elseif pair[2] == i
            push!(neighborIds, pair[1])
        end
    end
    return neighborIds
end

"""
    get_nodes_neighbor(instance::Instance, districts::Vector{Vector{Int}}, i::Int)

Retrieves nodes that are neighbors to a specific district.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts.
- `i::Int`: The index of the district to find neighbors for.

# Returns
- A list of nodes that are neighbors to the specified district.
"""

function get_nodes_neighbor(instance::Instance, districts::Vector{Vector{Int}}, i::Int)
    nodes = []
    for node in districts[i]
        for neighbor in neighbors(instance.graph, node)
            if neighbor in districts[i] ||
               neighbor == instance.num_blocks + 1 ||
               neighbor in nodes
                continue
            end
            push!(nodes, neighbor)
        end
    end
    return nodes
end

"""
    findSuitableNodeToAdd(instance::Instance, districts::Vector{Vector{Int}}, district_index::Int, neighborPairs::Dict{Tuple{Int, Int}, Int})

Finds a suitable node to add to a district that is under the minimum size limit.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts.
- `district_index::Int`: The index of the district needing an additional node.
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping pairs of neighboring districts to their ID.

# Returns
- The node to add and the district it belongs to, or (0, 0) if no suitable node is found.
"""

function findSuitableNodeToAdd(
    instance::Instance,
    districts::Vector{Vector{Int}},
    district_index::Int,
    neighborPairs::Dict{Tuple{Int,Int},Int},
)
    neighborIds = find_id_neighbors(district_index, neighborPairs)
    #order the neighbor districts by decreasing size
    neighborIds = sort(neighborIds, by = x -> length(districts[x]), rev = true)
    neighbor_nodes = get_nodes_neighbor(instance, districts, district_index)
    # if no suitable node is found, try with the next district
    for idx = 1:length(neighborIds)
        selected_district = neighborIds[idx]
        for node in districts[selected_district]
            if node in neighbor_nodes
                sub_graph, _ = induced_subgraph(
                    instance.graph,
                    setdiff(districts[selected_district], [node]),
                )
                if is_connected(sub_graph) &&
                   length(districts[selected_district]) > instance.min_district_size
                    return node, selected_district
                end
            end
        end
    end
    return 0, 0
end

"""
    findSuitableNodeToRemove(instance::Instance, districts::Vector{Vector{Int}}, district_index::Int, neighborPairs::Dict{Tuple{Int, Int}, Int})

Identifies a suitable node to remove from a district that is over the maximum size limit.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts.
- `district_index::Int`: The index of the district needing to shed a node.
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping pairs of neighboring districts to their ID.

# Returns
- The node to remove and the new district it should belong to, or (0, 0) if no suitable node is found.
"""

function findSuitableNodeToRemove(
    instance::Instance,
    districts::Vector{Vector{Int}},
    district_index::Int,
    neighborPairs::Dict{Tuple{Int,Int},Int},
)
    neighborIds = find_id_neighbors(district_index, neighborPairs)
    neighborIds = sort(neighborIds, by = x -> length(districts[x]))

    for idx = 1:length(neighborIds)
        selected_district = neighborIds[idx]
        neighbor_nodes = get_nodes_neighbor(instance, districts, selected_district)
        for node in districts[district_index]
            if node in neighbor_nodes
                sub_graph, _ = induced_subgraph(
                    instance.graph,
                    setdiff(districts[district_index], [node]),
                )
                if is_connected(sub_graph) &&
                   length(districts[selected_district]) < instance.max_district_size
                    return node, selected_district
                end
            end
        end
    end
    return 0, 0
end

"""
    repair_min_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districts that do not meet the minimum size requirement by adding nodes from neighboring districts.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be repaired.

# Returns
- The repaired list of districts.
"""

function repair_min_districts(instance::Instance, districts::Vector{Vector{Int}})
    districts = sort(districts, by = x -> length(x))
    neighborPairs = findNeighborDistricts(instance, districts)
    for i = 1:length(districts)
        if length(districts[i]) < instance.min_district_size
            while length(districts[i]) < instance.min_district_size
                node, neighbor_district =
                    findSuitableNodeToAdd(instance, districts, i, neighborPairs)
                if node == 0
                    break
                end
                push!(districts[i], node)
                districts[neighbor_district] = setdiff(districts[neighbor_district], [node])
                neighborPairs = findNeighborDistricts(instance, districts)
            end
        end
    end
    return districts
end

"""
    repair_max_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districts that exceed the maximum size limit by removing nodes to neighboring districts.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be repaired.

# Returns
- The repaired list of districts.
"""

function repair_max_districts(instance::Instance, districts::Vector{Vector{Int}})
    districts = sort(districts, by = x -> length(x), rev = true)
    neighborPairs = findNeighborDistricts(instance, districts)
    for i = 1:length(districts)
        if length(districts[i]) > instance.max_district_size
            while length(districts[i]) > instance.max_district_size
                node, new_district =
                    findSuitableNodeToRemove(instance, districts, i, neighborPairs)
                if node == 0
                    break
                end
                districts[i] = setdiff(districts[i], [node])
                push!(districts[new_district], node)
                neighborPairs = findNeighborDistricts(instance, districts)
            end
        end
    end
    return districts
end

"""
    repair_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districting solutions that do not respect the district size constraints. It adjusts districts to meet both minimum and maximum size requirements.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be repaired.

# Returns
- The repaired list of districts.
"""

function repair_districts(instance::Instance, districts::Vector{Vector{Int}})
    MAX_ITER = 10
    iter = 1
    is_valid_solution = is_valid_districting_solution(instance, districts)
    while !is_valid_solution && iter <= MAX_ITER
        districts = repair_max_districts(instance, districts)
        districts = repair_min_districts(instance, districts)
        iter += 1
    end
    return districts
end


"""
    is_valid_districting_solution(instance::Instance, districts::Vector{Vector{Int}})

Checks whether all districts are valid, i.e., they are within the maximum and minimum district size constraints.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be checked.

# Returns
- `True` if all districts are valid, and `False` if any district is invalid.
"""

function is_valid_districting_solution(instance::Instance, districts::Vector{Vector{Int}})
    for i = 1:length(districts)
        districtSize = length(districts[i])
        isTooLarge = districtSize > instance.max_district_size
        isTooSmall = districtSize < instance.min_district_size
        if isTooLarge || isTooSmall
            return false
        end
    end
    return true
end


"""
    initialize_solution(instance::Instance, costloader::Costloader, mode::String="CMST", model = nothing)

Creates an initial solution for the problem instance. The solution respects district size constraints and minimizes the total cost.

# Arguments
- `instance::Instance`: The problem instance.
- `costloader::Costloader`: The costloader object.
- `mode::String`: The mode of the solution (e.g., "CMST" or "Districting", etc.)

# Returns
- `solution::Solution`: The initial solution.
"""

function initialize_solution(
    instance::Instance,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)
    tree = run_kruskal(instance)
    districts = GreedyMerging(instance, tree)
    districts = repair_districts(instance, districts)
    return create_solution(instance, districts, costloader, mode, model)
end
