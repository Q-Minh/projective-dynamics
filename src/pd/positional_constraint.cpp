#include "pd/positional_constraint.h"

namespace pd {

positional_constraint_t::sparse_matrix_type
positional_constraint_t::get_Ai_Si(index_type vi, index_type N) const
{
    Eigen::SparseMatrix<scalar_type> Ai(3, 3);
    Ai.insert(0, 0) = scalar_type{1.};
    Ai.insert(1, 1) = scalar_type{1.};
    Ai.insert(2, 2) = scalar_type{1.};

    Eigen::SparseMatrix<scalar_type> Si(3, static_cast<std::size_t>(3u) * N);
    constexpr std::size_t three{3u};
    Si.insert(0, three * vi + 0) = scalar_type{1.};
    Si.insert(1, three * vi + 1) = scalar_type{1.};
    Si.insert(2, three * vi + 2) = scalar_type{1.};

    auto const AiSi = (Ai * Si).pruned();
    return AiSi;
}

positional_constraint_t::sparse_matrix_type positional_constraint_t::get_SiT_AiT_Bi() const
{
    sparse_matrix_type Bi(3, 3);
    Bi.insert(0, 0) = scalar_type{1.};
    Bi.insert(1, 1) = scalar_type{1.};
    Bi.insert(2, 2) = scalar_type{1.};

    return (Ai_Si_.transpose() * Bi).pruned();
}

void positional_constraint_t::project_wi_SiT_AiT_Bi_pi(q_type const& q, Eigen::VectorXd& b) const
{
    sparse_matrix_type const bi = wi() * SiT_AiT_Bi_ * p0_;
    for (int k = 0; k < bi.outerSize(); ++k)
        for (Eigen::SparseMatrix<scalar_type>::InnerIterator it(bi, k); it; ++it)
            b(it.row()) += it.value();
}

std::vector<Eigen::Triplet<positional_constraint_t::scalar_type>>
positional_constraint_t::get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const
{
    auto const vi = indices().at(0);
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