#ifndef PD_PD_POSITIONAL_CONSTRAINT_H
#define PD_PD_POSITIONAL_CONSTRAINT_H

#include "constraint.h"

namespace pd {

class positional_constraint_t : public constraint_t
{
  public:
    using self_type      = positional_constraint_t;
    using base_type      = constraint_t;
    using index_type     = std::uint32_t;
    using scalar_type    = double;
    using masses_type    = Eigen::VectorXd;
    using positions_type = typename base_type::positions_type;
    using position_type  = typename base_type::position_type;
    using gradient_type  = typename base_type::gradient_type;

  public:
    positional_constraint_t(std::initializer_list<index_type> indices, positions_type const& p)
        : base_type(indices)
    {
        assert(indices.size() == 1u);
    }
};

} // namespace pd

#endif // PD_PD_POSITIONAL_CONSTRAINT_H
