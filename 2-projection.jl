# Set up environment
import Pkg; Pkg.activate(".")

# Imports
using Distributions
using Plots
using Polynomials
using Optim
using ForwardDiff

# Setup
v_upper = 1.0
v_lower = 0.0
dist = Uniform(v_lower, v_upper)
N = 2

# Construct convenience functions
f0(x) = pdf(dist, x)
F0(x) = cdf(dist, x)

# ODE to solve
f(σ, v) = (N-1) * f0(v) / F0(v) * (v - σ)

# Solution setup
ϵ = 0.001
tspan = (v_lower + ϵ, v_upper - ϵ)
K = 20
T = K+1

"""
    estimate(θ)

`estimate` accepts a vector of length `N*K` and returns a `Tuple` containing

- `H`: The sum of square errors of the analytic derivative less the estimated function.
- `φ`: An `N` × `T` `Matrix` of the estimated inverse bid functions.
- `dφ`: An `N` × `T` `Matrix` of the estimated inverse bid function derivatives. 
- `dφ_true`: An `N` × `T` `Matrix` of the analytic derivative.
"""
function estimate(θ)
    α = reshape(θ, (N, K))
    # α = reshape(θ[2:end], (N, K))
    ts = collect(range(v_lower + ϵ, s_bar, length=T))
    φ = Array{eltype(θ), 2}(undef, N, T)
    dφ = Array{eltype(θ), 2}(undef, N, T)
    dφ_true = Array{eltype(θ), 2}(undef, N, T)

    for n in 1:N
        poly = Polynomial(α[n,:])
        φ[n,:] = poly.(ts)
        dφ[n,:] = (derivative(poly)).(ts)
        dφ_true[n,:] = f.(φ[n,:], ts)
    end

    H = sum((dφ - dφ_true) .^ 2)

    return H, φ, dφ, dφ_true
end

# Create the initial basis weighting matrix for each of the bidders.
# Setting this to start at an order-2 polynomial ensures that
# the optimizer can find a good starting point.
θ_init = zeros(K)
θ_init[1] = 1.0
θ_init[2] = 0.25

# Set up a target function to optimize on
target(x) = estimate(x)[1]
dtarget(G, x) = ForwardDiff.gradient(G, target, x)

# Optimize the ODE
opt_result = optimize(target, dtarget, θ_init, LBFGS())

# Check the optimization result
println(opt_result)

# Rerun the estimate function to get actual/estimated inverse bid functions
_, φ, dφ, dφ_exact = estimate(opt_result.minimizer)

# Plot the results
ts = collect(range(0, v_upper, length=T))

true_φ = map(x -> x/2, ts)
p1 = plot(ts, φ[1,:])
plot!(p1, ts, true_φ)

p2 = plot(ts, dφ[1,:])
plot!(p2, ts, dφ_exact[1,:])

plot(p1, p2)