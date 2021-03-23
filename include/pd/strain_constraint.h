#ifndef PD_PD_STRAIN_CONSTRAINT_H
#define PD_PD_STRAIN_CONSTRAINT_H

#include "constraint.h"

namespace pd {

class strain_constraint_t : public constraint_t
{
  public:
    using self_type          = strain_constraint_t;
    using base_type          = constraint_t;
    using index_type         = std::uint32_t;
    using scalar_type        = double;
    using masses_type        = Eigen::VectorXd;
    using positions_type     = typename base_type::positions_type;
    using q_type             = typename base_type::q_type;
    using gradient_type      = typename base_type::gradient_type;
    using sparse_matrix_type = Eigen::SparseMatrix<scalar_type>;

  public:
    strain_constraint_t(
        std::initializer_list<index_type> indices,
        scalar_type wi,
        positions_type const& p,
        scalar_type sigma_min,
        scalar_type sigma_max);

    virtual void project_wi_SiT_AiT_Bi_pi(q_type const& q, Eigen::VectorXd& b) const override;

    virtual std::vector<Eigen::Triplet<scalar_type>>
    get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const override;

  private:
    scalar_type V0_;
    Eigen::Matrix3d DmInv_;
    scalar_type sigma_min_;
    scalar_type sigma_max_;
};

} // namespace pd

#endif // PD_PD_STRAIN_CONSTRAINT_H