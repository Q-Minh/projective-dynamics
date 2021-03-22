#include "pd/edge_length_constraint.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <array>
#include <iostream>

namespace pd {
namespace detail {
namespace edge_length_constraint_ns {

/**
 * @brief
 * Ai = mean subtracted differential coordinate matrix
 * Si = selector matrix that selects positions of vi and vj from a vector q in R^3N
 * @param vi Index of particle i
 * @param vj Index of particle j
 * @param N Total number of particles
 * @return the product of matrices Ai*Si as a sparse matrix
 */
Eigen::SparseMatrix<edge_length_constraint_t::scalar_type> get_Ai_Si(int vi, int vj, int N)
{
    using scalar_type = typename edge_length_constraint_t::scalar_type;
    std::array<Eigen::Triplet<scalar_type>, 2u * 3u * 3u> AiSi_non_zero_entries{};

    // clang-format off
    // col 3*vi + 0
    AiSi_non_zero_entries[0]  = {0, 3*vi+0, static_cast<scalar_type>(/*1.0*/+0.5)};
    AiSi_non_zero_entries[1]  = {1, 3*vi+0, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[2]  = {2, 3*vi+0, static_cast<scalar_type>(/*0.0*/-0.5)};
    // col 3*vi + 1           
    AiSi_non_zero_entries[3]  = {0, 3*vi+1, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[4]  = {1, 3*vi+1, static_cast<scalar_type>(/*1.0*/+0.5)};
    AiSi_non_zero_entries[5]  = {2, 3*vi+1, static_cast<scalar_type>(/*0.0*/-0.5)};
    // col 3*vi + 2           
    AiSi_non_zero_entries[6]  = {0, 3*vi+2, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[7]  = {1, 3*vi+2, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[8]  = {2, 3*vi+2, static_cast<scalar_type>(/*1.0*/+0.5)};
    // col 3*vj + 0
    AiSi_non_zero_entries[9]  = {3, 3*vj+0, static_cast<scalar_type>(/*1.0*/+0.5)};
    AiSi_non_zero_entries[10] = {4, 3*vj+0, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[11] = {5, 3*vj+0, static_cast<scalar_type>(/*0.0*/-0.5)};
    // col 3*vj + 1
    AiSi_non_zero_entries[12] = {3, 3*vj+1, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[13] = {4, 3*vj+1, static_cast<scalar_type>(/*1.0*/+0.5)};
    AiSi_non_zero_entries[14] = {5, 3*vj+1, static_cast<scalar_type>(/*0.0*/-0.5)};
    // col 3*vj + 2
    AiSi_non_zero_entries[15] = {3, 3*vj+2, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[16] = {4, 3*vj+2, static_cast<scalar_type>(/*0.0*/-0.5)};
    AiSi_non_zero_entries[17] = {5, 3*vj+2, static_cast<scalar_type>(/*1.0*/+0.5)};
    // clang-format on

    Eigen::SparseMatrix<scalar_type> AiSi(6, 3 * N);
    AiSi.setFromTriplets(AiSi_non_zero_entries.begin(), AiSi_non_zero_entries.end());
    return AiSi;
}

} // namespace edge_length_constraint_ns
} // namespace detail

void edge_length_constraint_t::project_wi_SiT_AiT_Bi_pi(q_type const& q, Eigen::VectorXd& b) const
{
    auto const vi            = indices().at(0);
    auto const vj            = indices().at(1);
    Eigen::Vector3d const p1 = q.block(3 * vi, 0, 3, 1);
    Eigen::Vector3d const p2 = q.block(3 * vj, 0, 3, 1);
    auto const N             = q.rows() / 3;

    Eigen::SparseMatrix<scalar_type> AiSi(6, 3 * N);
    AiSi = detail::edge_length_constraint_ns::get_Ai_Si(vi, vj, N);

    using Bi_matrix_type = Eigen::Matrix<scalar_type, 6, 6>;
    Bi_matrix_type Bi;
    Bi.setIdentity();
    Eigen::Matrix3d const subtraction = scalar_type{0.5} * Eigen::Matrix<scalar_type, 3, 3>::Ones();
    Bi.block(0, 0, 3, 3) -= subtraction;
    Bi.block(3, 3, 3, 3) -= subtraction;

    // std::cout << "p1:\n" << p1 << "\n";
    // std::cout << "p2:\n" << p2 << "\n";

    Eigen::Vector3d const spring = p2 - p1;
    auto const length            = spring.norm();
    Eigen::Vector3d const n      = spring / length;
    auto const delta             = scalar_type{0.5} * (length - d_);

    // find the position p1 which results in ||p2 - p1|| = rest length
    Eigen::Vector3d const pi1 = p1 + delta * n;
    Eigen::Vector3d const pi2 = p2 - delta * n;

    using pi_vector_type = Eigen::Matrix<scalar_type, 6, 1>;
    pi_vector_type pi;
    pi(0) = pi1(0);
    pi(1) = pi1(1);
    pi(2) = pi1(2);
    pi(3) = pi2(0);
    pi(4) = pi2(1);
    pi(5) = pi2(2);

    pi_vector_type const pi_in_differential_coordinates = Bi * pi;

    // store Bi*pi as a sparse matrix so that the product (Ai*Si)^T * (Bi*Pi) is also a sparse
    // matrix
    Eigen::SparseMatrix<scalar_type> Bi_pi(6, 1);
    std::array<Eigen::Triplet<scalar_type>, 6u> Bi_pi_triplets{};
    Bi_pi_triplets[0] = {0, 0, pi_in_differential_coordinates(0)};
    Bi_pi_triplets[1] = {1, 0, pi_in_differential_coordinates(1)};
    Bi_pi_triplets[2] = {2, 0, pi_in_differential_coordinates(2)};
    Bi_pi_triplets[3] = {3, 0, pi_in_differential_coordinates(3)};
    Bi_pi_triplets[4] = {4, 0, pi_in_differential_coordinates(4)};
    Bi_pi_triplets[5] = {5, 0, pi_in_differential_coordinates(5)};
    Bi_pi.setFromTriplets(Bi_pi_triplets.begin(), Bi_pi_triplets.end());

    Eigen::SparseMatrix<scalar_type> const bi = wi() * (AiSi.transpose() * Bi_pi).pruned();

    // std::cout << "bi:\n" << bi << "\n";

    // update the right-hand side b vector
    for (int k = 0; k < bi.outerSize(); ++k)
        for (Eigen::SparseMatrix<scalar_type>::InnerIterator it(bi, k); it; ++it)
            b(it.row()) += it.value();
}

Eigen::VectorXd edge_length_constraint_t::project_wi_SiT_AiT_Bi_pi_dense(q_type const& q) const
{
    auto const vi            = indices().at(0);
    auto const vj            = indices().at(1);
    Eigen::Vector3d const p1 = q.block(3 * vi, 0, 3, 1);
    Eigen::Vector3d const p2 = q.block(3 * vj, 0, 3, 1);
    auto const N             = q.rows() / 3;

    Eigen::Matrix<scalar_type, 6, 6> Ai;
    Ai.setIdentity();

    Eigen::MatrixXd Si;
    Si.resize(6, 3 * N);
    Si.setZero();

    Si.block(0, 3 * vi, 3, 3) = Eigen::Matrix3d::Identity();
    Si.block(3, 3 * vj, 3, 3) = Eigen::Matrix3d::Identity();

    Eigen::MatrixXd const Ai_Si = Ai * Si;

    using Bi_matrix_type = Eigen::Matrix<scalar_type, 6, 6>;
    Bi_matrix_type Bi;
    Bi.setIdentity() /* - scalar_type{0.5} * Bi_matrix_type::Ones()*/;

    // std::cout << "p1:\n" << p1 << "\n";
    // std::cout << "p2:\n" << p2 << "\n";

    Eigen::Vector3d const spring = p2 - p1;
    auto const length            = spring.norm();
    Eigen::Vector3d const n      = spring / length;
    auto const delta             = scalar_type{0.5} * (length - d_);

    // find the position p1 which results in ||p2 - p1|| = rest length
    Eigen::Vector3d const pi1 = p1 + delta * n;
    Eigen::Vector3d const pi2 = p2 - delta * n;

    using pi_vector_type = Eigen::Matrix<scalar_type, 6, 1>;
    pi_vector_type pi;
    pi(0) = pi1(0);
    pi(1) = pi1(1);
    pi(2) = pi1(2);
    pi(3) = pi2(0);
    pi(4) = pi2(1);
    pi(5) = pi2(2);

    Eigen::Matrix<scalar_type, 6, 1> const Bi_pi = Bi * pi;
    Eigen::VectorXd const wi_SiT_AiT_Bi_pi       = wi() * Ai_Si.transpose() * Bi_pi;

    // std::cout << "Ai*Si:\n" << Ai_Si << "\n";
    // std::cout << "Bi*pi:\n" << Bi_pi << "\n";
    // std::cout << "wi * (Ai*Si)^T * Bi*pi:\n" << wi_SiT_AiT_Bi_pi << "\n";

    return wi_SiT_AiT_Bi_pi;
}

std::vector<Eigen::Triplet<edge_length_constraint_t::scalar_type>>
edge_length_constraint_t::get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const
{
    auto const vi = indices().at(0);
    auto const vj = indices().at(1);
    auto const N  = p.rows();

    Eigen::SparseMatrix<scalar_type> AiSi(6, 3 * N);
    AiSi = detail::edge_length_constraint_ns::get_Ai_Si(vi, vj, N);

    // std::cout << AiSi << "\n";

    // the product (Ai*Si)^T * (Ai*Si)
    Eigen::SparseMatrix<scalar_type> SiTAiTAiSi(3 * N, 3 * N);
    SiTAiTAiSi = wi() * (AiSi.transpose() * AiSi).pruned();

    // std::cout << SiTAiTAiSi << "\n";

    std::vector<Eigen::Triplet<scalar_type>> triplets_of_SiT_AiT_Ai_Si;
    triplets_of_SiT_AiT_Ai_Si.reserve(SiTAiTAiSi.nonZeros());

    for (int k = 0; k < SiTAiTAiSi.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<scalar_type>::InnerIterator it(SiTAiTAiSi, k); it; ++it)
        {
            int const i           = static_cast<int>(it.row());
            int const j           = static_cast<int>(it.col());
            scalar_type const aij = it.value();
            triplets_of_SiT_AiT_Ai_Si.push_back({i, j, aij});
        }
    }

    return triplets_of_SiT_AiT_Ai_Si;
}

Eigen::MatrixXd edge_length_constraint_t::get_wi_SiT_AiT_Ai_Si_dense(
    positions_type const& p,
    masses_type const& M) const
{
    auto const vi = indices().at(0);
    auto const vj = indices().at(1);
    auto const N  = p.rows();

    Eigen::Matrix<scalar_type, 6, 6> Ai;
    Ai.setIdentity();

    Eigen::MatrixXd Si;
    Si.resize(6, 3 * N);
    Si.setZero();

    Si.block(0, 3 * vi, 3, 3) = Eigen::Matrix3d::Identity();
    Si.block(3, 3 * vj, 3, 3) = Eigen::Matrix3d::Identity();

    Eigen::MatrixXd const Ai_Si            = Ai * Si;
    Eigen::MatrixXd const wi_SiT_AiT_Ai_Si = wi() * Ai_Si.transpose() * Ai_Si;

    // std::cout << "Ai*Si:\n" << Ai_Si << "\n";
    // std::cout << "wi * (Ai*Si)^T * (Ai*Si):\n" << wi_SiT_AiT_Ai_Si << "\n";

    return wi_SiT_AiT_Ai_Si;
}

} // namespace pd
