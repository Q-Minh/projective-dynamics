#include "pd/deformation_gradient_constraint.h"

#include <Eigen/Dense>
#include <eigen/SVD>

namespace pd {

deformation_gradient_constraint_t::deformation_gradient_constraint_t(
    std::initializer_list<index_type> indices,
    positions_type const& p,
    scalar_type young_modulus,
    scalar_type poisson_ratio)
    : base_type(indices), V0_{0.}, DmInv_{}, mu_{}, lambda_{}
{
    assert(indices.size() == 4u);

    auto const v1 = this->indices().at(0);
    auto const v2 = this->indices().at(1);
    auto const v3 = this->indices().at(2);
    auto const v4 = this->indices().at(3);

    auto const p1 = p.row(v1);
    auto const p2 = p.row(v2);
    auto const p3 = p.row(v3);
    auto const p4 = p.row(v4);

    Eigen::Matrix3d Dm;
    Dm.col(0) = (p1 - p4).transpose();
    Dm.col(1) = (p2 - p4).transpose();
    Dm.col(2) = (p3 - p4).transpose();

    V0_     = (1. / 6.) * Dm.determinant();
    DmInv_  = Dm.inverse();
    mu_     = (young_modulus) / (2. * (1 + poisson_ratio));
    lambda_ = (young_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio));
}

deformation_gradient_constraint_t::scalar_type
deformation_gradient_constraint_t::signed_volume(positions_type const& V) const
{
    Eigen::RowVector3d const p1 = V.row(indices()[0]);
    Eigen::RowVector3d const p2 = V.row(indices()[1]);
    Eigen::RowVector3d const p3 = V.row(indices()[2]);
    Eigen::RowVector3d const p4 = V.row(indices()[3]);

    Eigen::Matrix3d Ds;
    Ds.col(0)      = (p1 - p4).transpose();
    Ds.col(1)      = (p2 - p4).transpose();
    Ds.col(2)      = (p3 - p4).transpose();
    auto const vol = (1. / 6.) * Ds.determinant();
    return vol;
}

} // namespace pd