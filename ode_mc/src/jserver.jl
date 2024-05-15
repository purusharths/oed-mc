using UMBridge
using DynamicOED
using ModelingToolkit
using Optimization, OptimizationMOI, Ipopt


# Define the model behavior
function evaluate(input::Vector{Any}, config::Dict{String, Any})
    # Define the differential equations
    println("Input received: ", input)
    @variables t
    @variables x(t)=1.0 [description = "State"]
    @parameters p[1:1] = (input[1][1]*-1)/20 [description = "Fixed parameter", tunable = true]
    @variables obs(t) [description = "Observed", measurement_rate = 10]
    D = Differential(t)

    @named simple_system = ODESystem([
            D(x) ~ p[1] * x,
        ], tspan = (0.0, 1.0),
        observed = obs .~ [x.^2])

    @named oed = OEDSystem(simple_system)
    oed = structural_simplify(oed)

    # Augment the original problem to an OED problem
    oed_problem = OEDProblem(structural_simplify(oed), FisherACriterion())

    # Define an MTK Constraint system over the grid variables
    optimization_variables = states(oed_problem)
            
    constraint_equations = [
        sum(optimization_variables.measurements.w₁) ≲ 3,
    ]

    @named constraint_set = ConstraintsSystem(constraint_equations, optimization_variables, Num[])

    # Initialize the optimization problem
    optimization_problem = OptimizationProblem(oed_problem, AutoForwardDiff(),
        constraints = constraint_set,
        integer_constraints = false)

    # Solven for the optimal values of the observed variables
    println("Solving the optimization problem for observed variables.")
    optimal_design = solve(optimization_problem, Ipopt.Optimizer())
    u_opt = optimal_design.u + optimization_problem.u0
    println(u_opt)
    return [u_opt] 

end

testmodel = UMBridge.Model(name="forward", inputSizes=[1], outputSizes=[1])
UMBridge.define_evaluate(testmodel, evaluate)#(input, config) -> (2*input))
UMBridge.serve_models([testmodel], 4232)
