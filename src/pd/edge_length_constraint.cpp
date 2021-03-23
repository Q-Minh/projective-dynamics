#include "pd/edge_length_constraint.h"

#include <Eigen/SparseCore>
#include <array>

namespace pd {

void edge_length_constraint_t::project_wi_SiT_AiT_Bi_pi(q_type const& q, Eigen::VectorXd& b) const
{
    using index_type         = decltype(indices().front());
    index_type const vi      = indices().at(0);
    index_type const vj      = indices().at(1);
    Eigen::Vector3d const p1 = q.block(std::size_t{3u} * vi, 0, 3, 1);
    Eigen::Vector3d const p2 = q.block(std::size_t{3u} * vj, 0, 3, 1);
    auto const N             = q.rows() / 3;

    Eigen::Vector3d const spring = p2 - p1;
    auto const length            = spring.norm();
    Eigen::Vector3d const n      = spring / length;
    auto const delta             = scalar_type{0.5} * (length - d_);

    // find the position p1 which results in ||p2 - p1|| = rest length
    Eigen::Vector3d const pi1 = p1 + delta * n;
    Eigen::Vector3d const pi2 = p2 - delta * n;

    constexpr scalar_type half{0.5};
    constexpr std::size_t three{3};
    // the product wi * (Ai*Si)^T * (Ai*Si) only yields non-zero 
    // entries at coordinates [3vi, 3vi+3[ and [3vj, 3vj+3[.
    // The matrices Ai,Bi are differential coordinate matrices 
    // which result in mean subtraction in every dimension.
    // Thus, we subtract the mean in every dimension directly 
    // instead of performing the matrix multiplication.
    b(three * vi + 0) += wi() * half * (pi1.x() - pi2.x());
    b(three * vi + 1) += wi() * half * (pi1.y() - pi2.y());
    b(three * vi + 2) += wi() * half * (pi1.z() - pi2.z());

    b(three * vj + 0) += wi() * half * (pi2.x() - pi1.x());
    b(three * vj + 1) += wi() * half * (pi2.y() - pi1.y());
    b(three * vj + 2) += wi() * half * (pi2.z() - pi1.z());
}

std::vector<Eigen::Triplet<edge_length_constraint_t::scalar_type>>
edge_length_constraint_t::get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const
{
    int const vi = static_cast<int>(indices().at(0));
    int const vj = static_cast<int>(indices().at(1));
    auto const N = p.rows();

    // We precompute the product (Ai*Si)^T * (Ai*Si) and find that 
    // there are only nonzero elements in the blocks 
    // [3vi:3vi+3, 3vi:3vi+3], [3vj:3vj+3, 3vi:3vi+3],
    // [3vj:3vj+3, 3vi:3vi+3], [3vj:3vj+3, 3vj:3vj+3]
    // Those blocks contain the differential coordinates of 
    // the mean subtraction differential coordinates Ai.
    // We then multiply by wi as in wi * (Ai*Si)^T * (Ai*Si)
    constexpr scalar_type half{0.5};
    constexpr int three{3};
    std::array<Eigen::Triplet<scalar_type>, 12u> triplets;
    triplets[0] = {three * vi + 0, three * vi + 0, wi() * half};
    triplets[2] = {three * vi + 1, three * vi + 1, wi() * half};
    triplets[4] = {three * vi + 2, three * vi + 2, wi() * half};

    triplets[1] = {three * vj + 0, three * vi + 0, -wi() * half};
    triplets[3] = {three * vj + 1, three * vi + 1, -wi() * half};
    triplets[5] = {three * vj + 2, three * vi + 2, -wi() * half};

    triplets[6]  = {three * vi + 0, three * vj + 0, -wi() * half};
    triplets[8]  = {three * vi + 1, three * vj + 1, -wi() * half};
    triplets[10] = {three * vi + 2, three * vj + 2, -wi() * half};

    triplets[7]  = {three * vj + 0, three * vj + 0, wi() * half};
    triplets[9]  = {three * vj + 1, three * vj + 1, wi() * half};
    triplets[11] = {three * vj + 2, three * vj + 2, wi() * half};

    return std::vector<Eigen::Triplet<scalar_type>>{triplets.begin(), triplets.end()};
}

} // namespace pd
