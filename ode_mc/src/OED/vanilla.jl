using DynamicOED
using ModelingToolkit
using Optimization, OptimizationMOI, Ipopt

# Define the differential equations
println("Defining the differential equations")
@variables t
@variables x(t)=1.0 [description = "State"]
@parameters p[1:1]=-5.0 [description = "Fixed parameter", tunable = true]
@variables obs(t) [description = "Observed", measurement_rate = 10]
D = Differential(t)


println("Defining the ODE system")
@named simple_system = ODESystem([
        D(x) ~ p[1] * x,
    ], tspan = (0.0, 1.0),
    observed = obs .~ [x.^2])

@named oed = OEDSystem(simple_system)
oed = structural_simplify(oed)

# println(value(p[1]))
println("Defining the OED problem")
# Augment the original problem to an OED problem
oed_problem = OEDProblem(structural_simplify(oed), FisherACriterion())

# Define an MTK Constraint system over the grid variables
println("Defining the constraint system")
optimization_variables = states(oed_problem)
        
constraint_equations = [
      sum(optimization_variables.measurements.w₁) ≲ 3,
]

@named constraint_set = ConstraintsSystem(constraint_equations, optimization_variables, Num[])

# Initialize the optimization problem
println("Initializing the optimization problem")
optimization_problem = OptimizationProblem(oed_problem, AutoForwardDiff(),
      constraints = constraint_set,
      integer_constraints = false)

# Solven for the optimal values of the observed variables
println("Solving the optimization problem for observed variables.")
optimal_design = solve(optimization_problem, Ipopt.Optimizer())
u_opt = optimal_design.u + optimization_problem.u0