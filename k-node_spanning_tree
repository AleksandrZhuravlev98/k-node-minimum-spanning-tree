using JuMP, Gurobi, Graphs, LinearAlgebra, SparseArrays, ArgParse, LightGraphsFlows, Dates, DataFrames

function ILP_models(modelType::String, instance::String, relativeNrOfNodes::Float64)
    #==========================================================================#
    # DATA ====================================================================#
    #==========================================================================#

    # Note: the script is designed for .dat files that are used to build graphs.
    # Make sure that your files are compatible with the logic of the script. 

    # read data file
    dfile = open(instance)
    n = parse(Int64, readline(dfile)) # number of nodes
    m = parse(Int64, readline(dfile)) # number of edges

    source_ = zeros(Int64, 2 * m)
    target_ = zeros(Int64, 2 * m)
    weight_ = zeros(Int64, 2 * m)

    arc_index = spzeros(Int64, n, n) # sparse array to store arc indices for node pairs 

    for i=1:m
        str = readline(dfile)
        ind, u, v, w = split(str, " ")
        source_[i] = parse(Int64,u)+1
        target_[i] = parse(Int64,v)+1
        weight_[i] = parse(Int64,w)

        arc_index[source_[i],target_[i]] = i

        source_[i+m] = target_[i]
        target_[i+m] = source_[i]
        weight_[i+m] = weight_[i]
        arc_index[target_[i],source_[i]] = i+m         
    end

    dG = SimpleDiGraph(n) 
    for i = 1 : 2*m
        add_edge!(dG, source_[i],target_[i])
    end

    k = Int64(ceil((n-1)*relativeNrOfNodes))

    println("|V'| = ", n , ", |E| = ", m , "; k=", k)

        
    #==========================================================================#
    # MODEL SETUP =============================================================#
    #==========================================================================#

    k = k +1 # Increment k to account for the dummy node 

    model = direct_model(Gurobi.Optimizer())
    @variable(model, x[1:2*m], Bin) # is the edge used? 
    @variable(model, y[1:n], Bin) # y_i - indicator (is node i visited at all?)

    @objective(model, Min, sum(weight_[i]* x[i] for i in 1:2*m))
        
    #==========================================================================#
    # MTZ =====================================================================#
    #==========================================================================#
    start_time = now()
    if modelType == "mtz"
        println("MTZ formulation")
        
        # MTZ variable 
        @variable(model, u[1:n] >= 0) # u_i - order 

        # Constraint 1: Ensure that the root node has one outgoing edge 
        @constraint(model, sum(x[i] for i in 1:2*m if source_[i] == 1) == 1)

        # Constraint 1.1 And ensure that no node goes back to the initial node (it's a dummy!)
        @constraint(model, sum(x[i] for i in 1:2*m if target_[i] == 1) == 0)

        # Constraint 2: Ensure that the total number of selected edges is k-1
        #@constraint(model, sum(y[i] for i in 1:n) == k)
        @constraint(model, sum(x[i] for i in 1:2*m) == k-1)


        # Constraint 3: Each non-root node must have at most one incoming edge
        for j in 2:n
            @constraint(model, sum(x[i] for i in 1:2*m if target_[i] == j) <= 1)
        end

        # Constraint 4: Miller-Tucker-Zemlin (MTZ) Constraints to prevent cycles
        for i in 2:n
            @constraint(model, u[i] <= k-1)
            @constraint(model, u[i] >= 1)
        end

        for i in 1:2*m
            if source_[i] != target_[i] && source_[i] != 1 && target_[i] != 1
                @constraint(model, u[source_[i]] - u[target_[i]] + k * x[i] <= k - 1)
            end
        end

        # Constraint 5: If a node has an outgoing arc, it must have an incoming arc as well 
        for i in 2:n
            @constraint(model, sum(x[j] for j in 1:2*m if source_[j] == i) <= (k - 1) * y[i])
        end

        # Constraint 6: Also need to ensure that a node is marked as used (y=1) if the node has an incoming arc
        for i in 2:n
            @constraint(model, y[i] <= sum(x[j] for j in 1:2*m if target_[j] == i))
        end

        # Constraint 7: ... Because we constrain the root node to always be used (y=1)!
        @constraint(model, y[1] == 1)

        optimize!(model)
    #==========================================================================#
    # SCF =====================================================================#
    #==========================================================================#    
    elseif modelType == "scf"
        println("SCF formulation")
        
        @variable(model, f[1:2*m] >= 0) # flow variables

        # Constraint 1: Ensure that the root node has one outgoing edge
        @constraint(model, sum(x[i] for i in 1:2*m if source_[i] == 1) == 1)

        # Constraint 2: Ensure that no node goes back to the initial node (it's a dummy)
        @constraint(model, sum(x[i] for i in 1:2*m if target_[i] == 1) == 0)

        # Constraint 3: Ensure that the total number of selected nodes is k
        @constraint(model, sum(y[i] for i in 1:n) == k)

        # Constraint 4: Each non-root node must have at most one incoming edge
        for j in 2:n
            @constraint(model, sum(x[i] for i in 1:2*m if target_[i] == j) <= 1)
        end

        # Constraint 5: If a node has an outgoing arc, it must have an incoming arc as well 
        for i in 2:n
            @constraint(model, sum(x[j] for j in 1:2*m if source_[j] == i) <= (k - 1) * y[i])
        end

        # Constraint 6: Constrain the root node to always be used (y=1)
        @constraint(model, y[1] == 1)

        # Constraint 7: Ensure that a node is marked as used (y=1) if the node has an incoming arc
        for i in 2:n
            @constraint(model, y[i] <= sum(x[j] for j in 1:2*m if target_[j] == i))
        end

        # Constraint 8: The total flow out of the root node is k-1
        @constraint(model, sum(f[i] for i in 1:2*m if source_[i] == 1) == k-1)

        # Constraint 9: Flow conservation for all nodes except the root
        for j in 2:n
            @constraint(model, sum(f[i] for i in 1:2*m if target_[i] == j) - sum(f[i] for i in 1:2*m if source_[i] == j) == y[j])
        end

        # Constraint 10: Flow upper bound constrained by edge usage
        for i in 1:2*m
            @constraint(model, f[i] <= (k - 1) * x[i])
            @constraint(model, f[i] >= 0)
        end

        set_optimizer_attribute(model, "TimeLimit", 3600)

        # Optimize the model
        optimize!(model)
    
        
    elseif modelType == "mcf"
        println("MCF formulation")
        
        # Define the flow variables as a dictionary of SparseArrays

        f = Dict(mm => @variable(model, [1:2*m], lower_bound=0) for mm in 2:n)

        # Constraint 1: Ensure that the root node has one outgoing edge
        @constraint(model, sum(x[i] for i in 1:2*m if source_[i] == 1) == 1)

        # Constraint 2: Ensure that no node goes back to the initial node (it's a dummy)
        @constraint(model, sum(x[i] for i in 1:2*m if target_[i] == 1) == 0)

        # Constraint 3: Ensure that the total number of selected nodes is k
        @constraint(model, sum(y[i] for i in 1:n) == k)

        # Constraint 4: Each non-root node must have at most one incoming edge
        for j in 2:n
            @constraint(model, sum(x[i] for i in 1:2*m if target_[i] == j) <= 1)
        end

        # Constraint 5: If a node has an outgoing arc, it must have an incoming arc as well 
        for i in 2:n
            @constraint(model, sum(x[j] for j in 1:2*m if source_[j] == i) <= (k - 1) * y[i])
        end

        # Constraint 6: Constrain the root node to always be used (y=1)
        @constraint(model, y[1] == 1)

        # Constraint 7: Ensure that a node is marked as used (y=1) if the node has an incoming arc
        for i in 2:n
            @constraint(model, y[i] <= sum(x[j] for j in 1:2*m if target_[j] == i))
        end

    
        # Constraint 8: The total flow out of the root node is 1 for each commodity
        for mm in 2:n
            @constraint(model, sum(f[mm][i] for i in 1:2*m if source_[i] == 1) == y[mm])
        end

        # Constraint 9: For each node m, the sum of the corresponding commodity flows is always equal to 1
        
        for mm in 2:n
            @constraint(model, sum(f[mm][i] for i in 1:2*m if target_[i]==mm) == y[mm])
        end

        # Constraint 10: The net flow is zero for each node for eacvh commodity as long as the node is used

        for mm in 2:n
            for j in 2:n
                if j!=mm
                @constraint(model, (sum(f[mm][i] for i in 1:2*m if target_[i]==j && target_[i]!=source_[i]) -sum(f[mm][i] for i in 1:2*m if source_[i]==j && target_[i]!=source_[i])) == 0)        
                end
            end
        end


        # Constraint 11: Ensure connectivity and bound flow by edge usage
        for mm in 2:n
            for i in 1:2*m
                if target_[i] != source_[i]
                @constraint(model, f[mm][i] <= x[i])
                @constraint(model, f[mm][i] >= 0)
                end
            end
        end

            # Optimize the model
            optimize!(model)    
    
    else
        print("ERROR: Unknown model type!")
    end


    
    end_time = now()
    execution_time = end_time - start_time
    number_of_variables = num_variables(model)
    number_of_constraints = sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model))
    objective_val = objective_value(model)

    # Create a df for results 
    results = DataFrame(Instance = [instance], modelType = [modelType],
                        V = [n], 
                        fraction_used = [relativeNrOfNodes],
                        k = [k],
                        number_of_variables = [number_of_variables],
                        number_of_constraints = [number_of_constraints],
                        execution_time = [execution_time],
                        objective_value = [objective_val])

    return results
end


inst_list = ["g01.dat", "g04.dat"] # Be  careful - the thing doesn't work for very large graphs! 
model_list = ["mtz"]
node_fractions = [0.2, 0.5]

# Initialise an empty DataFrame to store results
all_results = DataFrame()

# Iterate over all combinations of parameters
for instance in inst_list
    for model in model_list
        for fraction in node_fractions
            result = ILP_models(model, instance, fraction)
            all_results = vcat(all_results, result)
        end
    end
end

# Save the final DataFrame as a CSV file
#CSV.write("MSTP_iteration_results - ALL.csv", all_results)
