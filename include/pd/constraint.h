#ifndef PD_PD_CONSTRAINT_H
#define PD_PD_CONSTRAINT_H

#include <Eigen/Core>
#include <vector>

namespace pd {

class constraint_t
{
  public:
    using index_type     = std::uint32_t;
    using masses_type    = Eigen::VectorXd;
    using positions_type = Eigen::MatrixXd;
    using position_type  = Eigen::RowVector3d;
    using gradient_type  = Eigen::Vector3d;
    using scalar_type    = double;

  public:
    constraint_t(std::initializer_list<index_type> indices) : indices_(indices) {}

    std::vector<index_type> const& indices() const { return indices_; }

  private:
    std::vector<index_type> indices_;
};

} // namespace pd

#endif // PD_PD_CONSTRAINT_H