#ifndef PBD_SOLVE_H
#define PBD_SOLVE_H

#include "pd/deformable_mesh.h"

namespace solver {

void solve(
    pd::deformable_mesh_t& model,
    Eigen::MatrixX3d const& fext,
    double dt                = 0.01,
    std::uint32_t iterations = 10,
    std::uint32_t substeps   = 10);

} // namespace solver

#endif // PBD_SOLVE_H
