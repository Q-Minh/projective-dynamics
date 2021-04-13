#ifndef PD_PD_CONSTRAINT_H
#define PD_PD_CONSTRAINT_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>

namespace pd {

class constraint_t
{
  public:
    using index_type     = std::uint32_t;
    using masses_type    = Eigen::VectorXd;
    using positions_type = Eigen::MatrixXd;
    using q_type         = Eigen::VectorXd;
    using position_type  = Eigen::RowVector3d;
    using gradient_type  = Eigen::Vector3d;
    using scalar_type    = double;

  public:
    constraint_t(std::initializer_list<index_type> indices)
        : indices_(indices), wi_(scalar_type{1.})
    {
    }

    constraint_t(std::initializer_list<index_type> indices, scalar_type wi)
        : indices_(indices), wi_(wi)
    {
    }

    virtual scalar_type evaluate(positions_type const& p, masses_type const& M)
    {
        return scalar_type{0.};
    }

    std::vector<index_type> const& indices() const { return indices_; }
    scalar_type wi() const { return wi_; }

    virtual void project_wi_SiT_AiT_Bi_pi(q_type const& q, Eigen::VectorXd& rhs) const = 0;
    virtual std::vector<Eigen::Triplet<scalar_type>>
    get_wi_SiT_AiT_Ai_Si(positions_type const& p, masses_type const& M) const = 0;

  private:
    std::vector<index_type> indices_;
    scalar_type wi_;
};

} // namespace pd

#endif // PD_PD_CONSTRAINT_H