module DECARBGurobiExt

using DECARB
using Gurobi

DECARB.gurobi_optimizer(::DECARB.GurobiBackend) = Gurobi.Optimizer

end
