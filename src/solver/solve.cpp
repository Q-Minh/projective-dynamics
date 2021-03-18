#include "solver/solve.h"

namespace solver {

void solve(
    pd::deformable_mesh_t& model,
    Eigen::MatrixX3d const& fext,
    double timestep,
    std::uint32_t iterations,
    std::uint32_t substeps)
{
    auto const num_iterations = iterations / substeps;
    double dt                 = timestep / static_cast<double>(substeps);
    auto const& constraints   = model.constraints();
    auto const J              = constraints.size();

    for (auto s = 0u; s < substeps; ++s)
    {
    }
}

} // namespace solver
