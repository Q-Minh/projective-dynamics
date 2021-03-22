//#ifndef PD_PD_DEFORMATION_GRADIENT_CONSTRAINT_H
//#define PD_PD_DEFORMATION_GRADIENT_CONSTRAINT_H
//
//#include "constraint.h"
//
//namespace pd {
//
//class deformation_gradient_constraint_t : public constraint_t
//{
//  public:
//    using self_type      = deformation_gradient_constraint_t;
//    using base_type      = constraint_t;
//    using index_type     = std::uint32_t;
//    using scalar_type    = typename constraint_t::scalar_type;
//    using masses_type    = Eigen::VectorXd;
//    using positions_type = typename base_type::positions_type;
//    using gradient_type  = typename base_type::gradient_type;
//
//  public:
//    deformation_gradient_constraint_t(
//        std::initializer_list<index_type> indices,
//        positions_type const& p,
//        scalar_type young_modulus,
//        scalar_type poisson_ratio);
//
//  protected:
//    scalar_type signed_volume(positions_type const& V) const;
//
//  private:
//    scalar_type V0_;
//    Eigen::Matrix3d DmInv_;
//    scalar_type mu_;
//    scalar_type lambda_;
//};
//
//} // namespace pd
//
//#endif // PD_PD_DEFORMATION_GRADIENT_CONSTRAINT_H