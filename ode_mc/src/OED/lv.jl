using DynamicOED
using ModelingToolkit
using Optimization, OptimizationMOI, Ipopt
using Plots

@variables t
@variables x(t)=0.5 [description = "Biomass Prey"] y(t)=0.7 [description ="Biomass Predator"]
@variables u(t) [description = "Control"]
@parameters p[1:2]=[1.0; 1.0] [description = "Fixed Parameters", tunable = false]
@parameters p_est[1:2]=[1.0; 1.0] [description = "Tunable Parameters", tunable = true]
D = Differential(t)
@variables obs(t)[1:2] [description = "Observed", measurement_rate = 96]
obs = collect(obs)

@named lotka_volterra = ODESystem(
    [
        D(x) ~   p[1]*x - p_est[1]*x*y;
        D(y) ~  -p[2]*y + p_est[2]*x*y
    ], tspan = (0.0, 12.0),
    observed = obs .~ [x; y]
)

@named oed_system = OEDSystem(lotka_volterra)

oed_problem = OEDProblem(structural_simplify(oed_system), DCriterion())

optimization_variables = states(oed_problem)

constraint_equations = [
    sum(optimization_variables.measurements.w₁) ≲ 32,
    sum(optimization_variables.measurements.w₂) ≲ 32,
]
println("Constraint Equations: ", constraint_equations)
@named constraint_system = ConstraintsSystem(
    constraint_equations, optimization_variables, []
)
println("Constraint System: ", constraint_system)

optimization_problem = OptimizationProblem(
    oed_problem, AutoForwardDiff(), constraints = constraint_system,
    integer_constraints = false
)

optimal_design = solve(optimization_problem, Ipopt.Optimizer(); hessian_approximation="limited-memory")

u_opt = optimal_design.u + optimization_problem.u0
println(u_opt)

function plotoed(problem, res)

    predictor = DynamicOED.build_predictor(problem)
    x_opt, t_opt = predictor(res)
    timegrid = problem.timegrid

    state_plot = plot(t_opt, x_opt[1:2, :]', xlabel = "Time", ylabel = "States", label = ["x" "y"])

    measures_plot = plot()
    for i in 1:2
        t_measures = vcat(first.(timegrid.timegrids[i]), last.(timegrid.timegrids[i]))
        sort!(t_measures)
        unique!(t_measures)
        _measurements = getfield(res.measurements |> NamedTuple, timegrid.variables[i])
        plot!(t_measures,
            vcat(_measurements, last(_measurements)),
            line = :steppost,
            xlabel = "Time",
            ylabel = "Measurement",
            color = i == 2 ? :red : :blue,
            label = string(timegrid.variables[i]))
    end

    plot(state_plot, measures_plot, layout=(2,1))
end

plotoed(oed_problem, u_opt)