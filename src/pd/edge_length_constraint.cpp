#include "pd/edge_length_constraint.h"

namespace pd {

edge_length_constraint_t::scalar_type
edge_length_constraint_t::evaluate(positions_type const& p, masses_type const& M) const
{
    auto const v0 = indices().at(0);
    auto const v1 = indices().at(1);
    auto const p0 = p.row(v0);
    auto const p1 = p.row(v1);

    return (p0 - p1).norm() - d_;
}

} // namespace pd
