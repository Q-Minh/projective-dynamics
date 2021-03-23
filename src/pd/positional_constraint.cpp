#include "pd/positional_constraint.h"

#include <array>

namespace pd {

void positional_constraint_t::project_wi_SiT_AiT_Bi_pi(q_type const& q, Eigen::VectorXd& b) const
{
    // Ai = identity3x3, Bi = identity3x3, Si = zeros3x3N + identity3x3 at block(3*vi, 0, 3, 3)
    // We precompute the non-zero entries of (Ai*Si)^T * (Bi*pi) which only occur 
    // at indices 3*vi + 0, 3*vi + 1 and 3*vi + 2 in the b vector. We then multiply 
    // by wi as in wi * (Ai*Si)^T * (Bi*pi). With positional constraints, our pi 
    // is simply the goal position p0.
    std::size_t const vi = static_cast<std::size_t>(indices().at(0));
    std::size_t constexpr three{3u};
    b.block(three * vi, 0, 3, 1) += wi() * p0_;
}

std::vector<Eigen::Triplet<positional_constraint_t::scalar_type>>
positional_constraint_t::get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const
{
    int const vi = static_cast<int>(indices().at(0));

    // Ai = identity3x3, Si = zeros3x3N + identity3x3 at block(3*vi, 0, 3, 3)
    // the computation (Ai*Si)^T * (Ai*Si) is precomputed and yields
    // a 3Nx3N matrix with an identity block at block(3*vi, 3*vi, 3, 3).
    // We multiply this identity block by wi
    std::array<Eigen::Triplet<scalar_type>, 3u> triplets;
    triplets[0] = {3 * vi + 0, 3 * vi + 0, wi()};
    triplets[1] = {3 * vi + 1, 3 * vi + 1, wi()};
    triplets[2] = {3 * vi + 2, 3 * vi + 2, wi()};

    return std::vector<Eigen::Triplet<scalar_type>>{triplets.begin(), triplets.end()};
}

} // namespace pd