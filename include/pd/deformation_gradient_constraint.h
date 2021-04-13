#ifndef PD_PD_DEFORMATION_GRADIENT_CONSTRAINT_H
#define PD_PD_DEFORMATION_GRADIENT_CONSTRAINT_H

#include "constraint.h"

namespace pd {

class deformation_gradient_constraint_t : public constraint_t
{
  public:
    using self_type          = deformation_gradient_constraint_t;
    using base_type          = constraint_t;
    using index_type         = std::uint32_t;
    using scalar_type        = double;
    using masses_type        = Eigen::VectorXd;
    using positions_type     = typename base_type::positions_type;
    using q_type             = typename base_type::q_type;
    using gradient_type      = typename base_type::gradient_type;
    using sparse_matrix_type = Eigen::SparseMatrix<scalar_type>;

  public:
    deformation_gradient_constraint_t(
        std::initializer_list<index_type> indices,
        scalar_type wi,
        positions_type const& p);

    virtual scalar_type evaluate(positions_type const& p, masses_type const& M) override;

    virtual void project_wi_SiT_AiT_Bi_pi(q_type const& q, Eigen::VectorXd& rhs) const override;

    virtual std::vector<Eigen::Triplet<scalar_type>>
    get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const override;

  private:
    scalar_type V0_;
    Eigen::Matrix3d DmInv_;
};

} // namespace pd

#endif // PD_PD_DEFORMATION_GRADIENT_CONSTRAINT_H