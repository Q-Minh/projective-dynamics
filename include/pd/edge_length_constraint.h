#ifndef PD_PD_EDGE_LENGTH_CONSTRAINT_H
#define PD_PD_EDGE_LENGTH_CONSTRAINT_H

#include "constraint.h"

namespace pd {

class edge_length_constraint_t : public constraint_t
{
  public:
    using self_type          = edge_length_constraint_t;
    using base_type          = constraint_t;
    using index_type         = std::uint32_t;
    using scalar_type        = double;
    using masses_type        = Eigen::VectorXd;
    using positions_type     = typename base_type::positions_type;
    using q_type             = typename base_type::q_type;
    using gradient_type      = typename base_type::gradient_type;
    using sparse_matrix_type = Eigen::SparseMatrix<scalar_type>;

  public:
    edge_length_constraint_t(
        std::initializer_list<index_type> indices,
        scalar_type wi,
        positions_type const& p)
        : base_type(indices, wi), d_(0.)
    {
        assert(indices.size() == 2u);
        auto const e0 = this->indices()[0];
        auto const e1 = this->indices()[1];

        d_ = (p.row(e0) - p.row(e1)).norm();
    }

    virtual void project_wi_SiT_AiT_Bi_pi(q_type const& q, Eigen::VectorXd& rhs) const override;

    virtual std::vector<Eigen::Triplet<scalar_type>>
    get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const override;

  private:
    scalar_type d_; ///< rest length
    sparse_matrix_type Ai_Si_;
    sparse_matrix_type SiT_AiT_Bi_;
};

} // namespace pd

#endif // PD_PD_EDGE_LENGTH_CONSTRAINT_H
