//#include "pd/tetrahedron_volume_constraint.h"
//
//#include <Eigen/Dense>
//
//namespace pd {
//
//tetrahedron_volume_constraint_t::tetrahedron_volume_constraint_t(
//    std::initializer_list<index_type> indices,
//    positions_type const& p)
//    : base_type(indices), V0_{0.}
//{
//    assert(indices.size() == 4u);
//    V0_ = volume(p);
//}
//
//tetrahedron_volume_constraint_t::scalar_type
//tetrahedron_volume_constraint_t::volume(positions_type const& V) const
//{
//    Eigen::RowVector3d const p0 = V.row(indices()[0]);
//    Eigen::RowVector3d const p1 = V.row(indices()[1]);
//    Eigen::RowVector3d const p2 = V.row(indices()[2]);
//    Eigen::RowVector3d const p3 = V.row(indices()[3]);
//
//    auto const vol = (1. / 6.) * (p1 - p0).cross(p2 - p0).dot(p3 - p0);
//    return std::abs(vol);
//}
//
//tetrahedron_volume_constraint_t::scalar_type
//tetrahedron_volume_constraint_t::evaluate(positions_type const& p, masses_type const& m) const
//{
//    return volume(p) - V0_;
//}
//
//} // namespace pd