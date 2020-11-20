# Set up environment
import Pkg; Pkg.activate(".")

# Imports
using DifferentialEquations
using ModelingToolkit
using Distributions
using Plots

# Setup
v_upper = 1.0
v_lower = 0.0
dist = Uniform(v_lower, v_upper)
n = 2

# Construct convenience functions
f0(x) = pdf(dist, x)
F0(x) = cdf(dist, x)

# ODE to solve
f(σ, p, v) = (n-1) * f0(v) / F0(v) * (v - σ)

# Solution setup
ϵ = 0.001
tspan = (v_lower + ϵ, v_upper - ϵ)
u0 = 0

# Solve the ODE
problem = ODEProblem(f, u0, tspan)
solution = solve(problem, Tsit5(), reltol=1e-8, abstol=1e-8)

# Extract the solution
ts = solution.t
us = solution.u

# Calculate the analytic solution
σ_analytic(v) = v / 2
σs = σ_analytic.(ts)

# Plot it
plot(solution)
plot!(ts, σs)
