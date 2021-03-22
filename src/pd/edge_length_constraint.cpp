#include "pd/edge_length_constraint.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <array>
#include <iostream>

namespace pd {

edge_length_constraint_t::sparse_matrix_type
edge_length_constraint_t::get_Ai_Si(int vi, int vj, int N) const
{
    using scalar_type = typename edge_length_constraint_t::scalar_type;
    std::array<Eigen::Triplet<scalar_type>, 4u * 3u> AiSi_non_zero_entries{};

    // clang-format off
    // col 3*vi + 0
    AiSi_non_zero_entries[0]  = {0, 3*vi+0, static_cast<scalar_type>(/*1.0*/+0.5)};
    AiSi_non_zero_entries[1]  = {3, 3*vi+0, static_cast<scalar_type>(/*0.0*/-0.5)};
    // col 3*vi + 1           
    AiSi_non_zero_entries[2]  = {1, 3*vi+1, static_cast<scalar_type>(/*0.0*/+0.5)};
    AiSi_non_zero_entries[3]  = {4, 3*vi+1, static_cast<scalar_type>(/*1.0*/-0.5)};
    // col 3*vi + 2           
    AiSi_non_zero_entries[4]  = {2, 3*vi+2, static_cast<scalar_type>(/*0.0*/+0.5)};
    AiSi_non_zero_entries[5]  = {5, 3*vi+2, static_cast<scalar_type>(/*0.0*/-0.5)};
    // col 3*vj + 0
    AiSi_non_zero_entries[6]  = {0, 3*vj+0, static_cast<scalar_type>(/*1.0*/-0.5)};
    AiSi_non_zero_entries[7]  = {3, 3*vj+0, static_cast<scalar_type>(/*0.0*/+0.5)};
    // col 3*vj + 1
    AiSi_non_zero_entries[8]  = {1, 3*vj+1, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[9]  = {4, 3*vj+1, static_cast<scalar_type>(/*1.0*/+0.5)};
    // col 3*vj + 2
    AiSi_non_zero_entries[10] = {2, 3*vj+2, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[11] = {5, 3*vj+2, static_cast<scalar_type>(/*0.0*/+0.5)};
    // clang-format on

    Eigen::SparseMatrix<scalar_type> AiSi(6, 3 * N);
    AiSi.setFromTriplets(AiSi_non_zero_entries.begin(), AiSi_non_zero_entries.end());
    return AiSi;
}

edge_length_constraint_t::sparse_matrix_type
edge_length_constraint_t::get_SiT_AiT_Bi(int vi, int vj) const
{
    std::array<Eigen::Triplet<scalar_type>, 12u> triplets;
    triplets[0] = {0, 0, scalar_type{+0.5}};
    triplets[1] = {0, 3, scalar_type{-0.5}};
    triplets[2] = {3, 0, scalar_type{-0.5}};
    triplets[3] = {3, 3, scalar_type{+0.5}};

    triplets[4] = {1, 1, scalar_type{+0.5}};
    triplets[5] = {1, 4, scalar_type{-0.5}};
    triplets[6] = {4, 1, scalar_type{-0.5}};
    triplets[7] = {4, 4, scalar_type{+0.5}};

    triplets[8]  = {2, 2, scalar_type{+0.5}};
    triplets[9]  = {2, 5, scalar_type{-0.5}};
    triplets[10] = {5, 2, scalar_type{-0.5}};
    triplets[11] = {5, 5, scalar_type{+0.5}};

    sparse_matrix_type Bi(6, 6);
    Bi.setFromTriplets(triplets.begin(), triplets.end());
    return (Ai_Si_.transpose() * Bi).pruned();
}

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

    sparse_matrix_type pi(6, 1);
    pi.insert(0, 0) = pi1(0);
    pi.insert(1, 0) = pi1(1);
    pi.insert(2, 0) = pi1(2);
    pi.insert(3, 0) = pi2(0);
    pi.insert(4, 0) = pi2(1);
    pi.insert(5, 0) = pi2(2);

    sparse_matrix_type const bi = wi() * (SiT_AiT_Bi_ * pi);

    // update the right-hand side b vector
    for (int k = 0; k < bi.outerSize(); ++k)
        for (Eigen::SparseMatrix<scalar_type>::InnerIterator it(bi, k); it; ++it)
            b(it.row()) += it.value();
}

std::vector<Eigen::Triplet<edge_length_constraint_t::scalar_type>>
edge_length_constraint_t::get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const
{
    auto const vi = indices().at(0);
    auto const vj = indices().at(1);
    auto const N  = p.rows();

    sparse_matrix_type const wi_SiT_AiT_Ai_Si = wi() * Ai_Si_.transpose() * Ai_Si_;

    std::vector<Eigen::Triplet<scalar_type>> triplets_of_SiT_AiT_Ai_Si;
    triplets_of_SiT_AiT_Ai_Si.reserve(wi_SiT_AiT_Ai_Si.nonZeros());

    for (int k = 0; k < wi_SiT_AiT_Ai_Si.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<scalar_type>::InnerIterator it(wi_SiT_AiT_Ai_Si, k); it; ++it)
        {
            int const i           = static_cast<int>(it.row());
            int const j           = static_cast<int>(it.col());
            scalar_type const aij = it.value();
            triplets_of_SiT_AiT_Ai_Si.push_back({i, j, aij});
        }
    }

    return triplets_of_SiT_AiT_Ai_Si;
}

} // namespace pd
