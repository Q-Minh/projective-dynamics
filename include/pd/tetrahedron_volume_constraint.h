//#ifndef PD_PD_TETRAHEDRON_VOLUME_CONSTRAINT_H
//#define PD_PD_TETRAHEDRON_VOLUME_CONSTRAINT_H
//
//#include "constraint.h"
//
//namespace pd {
//
//class tetrahedron_volume_constraint_t : public constraint_t
//{
//  public:
//    using self_type      = tetrahedron_volume_constraint_t;
//    using base_type      = constraint_t;
//    using index_type     = std::uint32_t;
//    using scalar_type    = double;
//    using masses_type    = Eigen::VectorXd;
//    using positions_type = typename base_type::positions_type;
//    using gradient_type  = typename base_type::gradient_type;
//
//  public:
//    tetrahedron_volume_constraint_t(
//        std::initializer_list<index_type> indices,
//        positions_type const& p);
//
//    scalar_type evaluate(positions_type const& V, masses_type const& M) const;
//
//  protected:
//    scalar_type volume(positions_type const& V) const;
//
//  private:
//    scalar_type V0_;
//};
//
//} // namespace pd
//
//#endif // PD_PD_TETRAHEDRON_VOLUME_CONSTRAINT_H
