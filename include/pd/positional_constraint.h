#ifndef PD_PD_POSITIONAL_CONSTRAINT_H
#define PD_PD_POSITIONAL_CONSTRAINT_H

#include "constraint.h"

namespace pd {

class positional_constraint_t : public constraint_t
{
  public:
    using self_type          = positional_constraint_t;
    using base_type          = constraint_t;
    using index_type         = std::uint32_t;
    using scalar_type        = double;
    using masses_type        = Eigen::VectorXd;
    using positions_type     = typename base_type::positions_type;
    using q_type             = typename base_type::q_type;
    using gradient_type      = typename base_type::gradient_type;
    using sparse_matrix_type = Eigen::SparseMatrix<scalar_type>;

  public:
    positional_constraint_t(
        std::initializer_list<index_type> indices,
        scalar_type wi,
        positions_type const& p)
        : base_type(indices, wi), p0_(3, 1)
    {
        assert(indices.size() == 1u);
        auto const vi = this->indices().front();
        auto const N  = p.rows();
        p0_           = p.row(vi).transpose();
    }

    virtual void project_wi_SiT_AiT_Bi_pi(q_type const& q, Eigen::VectorXd& rhs) const override;
    virtual std::vector<Eigen::Triplet<scalar_type>>
    get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const override;

  private:
    Eigen::Vector3d p0_;
};

} // namespace pd

#endif // PD_PD_POSITIONAL_CONSTRAINT_H
