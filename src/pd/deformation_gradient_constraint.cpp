#include "pd/deformation_gradient_constraint.h"

#include <Eigen/SVD>
#include <array>

namespace pd {

deformation_gradient_constraint_t::deformation_gradient_constraint_t(
    std::initializer_list<index_type> indices,
    scalar_type wi,
    positions_type const& p)
    : base_type(indices, wi), V0_{0.}, DmInv_{}
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

    V0_    = (1. / 6.) * Dm.determinant();
    DmInv_ = Dm.inverse();
}

void deformation_gradient_constraint_t::project_wi_SiT_AiT_Bi_pi(
    q_type const& q,
    Eigen::VectorXd& b) const
{
    auto const N  = q.rows() / 3;
    auto const v1 = this->indices().at(0);
    auto const v2 = this->indices().at(1);
    auto const v3 = this->indices().at(2);
    auto const v4 = this->indices().at(3);

    std::size_t const vi = static_cast<std::size_t>(3u) * v1;
    std::size_t const vj = static_cast<std::size_t>(3u) * v2;
    std::size_t const vk = static_cast<std::size_t>(3u) * v3;
    std::size_t const vl = static_cast<std::size_t>(3u) * v4;

    Eigen::Vector3d const q1 = q.block(vi, 0, 3, 1);
    Eigen::Vector3d const q2 = q.block(vj, 0, 3, 1);
    Eigen::Vector3d const q3 = q.block(vk, 0, 3, 1);
    Eigen::Vector3d const q4 = q.block(vl, 0, 3, 1);

    Eigen::Matrix3d Ds;
    Ds.col(0) = q1 - q4;
    Ds.col(1) = q2 - q4;
    Ds.col(2) = q3 - q4;

    Eigen::Matrix3d const F = Ds * DmInv_;
    // scalar_type const Vol     = (1. / 6.) * Ds.determinant();
    // bool const is_V_positive  = Vol >= scalar_type{0.};
    // bool const is_V0_positive = V0_ >= scalar_type{0.};

    // TODO: tet inversion handling?
    // bool const is_tet_inverted =
    //    (is_V_positive && !is_V0_positive) || (!is_V_positive && is_V0_positive);

    Eigen::JacobiSVD<Eigen::Matrix3d> SVD(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d const& U = SVD.matrixU();
    Eigen::Matrix3d const& V = SVD.matrixV();

    Eigen::Matrix3d R = U * V.transpose();
    if (R.determinant() < 0)
    {
        R.col(2) = -R.col(2);
    }

    auto const w         = this->wi();
    scalar_type const V0 = std::abs(V0_);
    auto const weight    = w * V0;

    // Uncomment to use sparse matrix products to compute right hand side
    /*static sparse_matrix_type Bi(9, 9);
    if (Bi.nonZeros() == 0)
    {
        Bi.insert(0, 0) = scalar_type{1.};
        Bi.insert(1, 1) = scalar_type{1.};
        Bi.insert(2, 2) = scalar_type{1.};
        Bi.insert(3, 3) = scalar_type{1.};
        Bi.insert(4, 4) = scalar_type{1.};
        Bi.insert(5, 5) = scalar_type{1.};
        Bi.insert(6, 6) = scalar_type{1.};
        Bi.insert(7, 7) = scalar_type{1.};
        Bi.insert(8, 8) = scalar_type{1.};
    }

    sparse_matrix_type pi(9, 1);
    pi.insert(0, 0) = R(0, 0);
    pi.insert(1, 0) = R(1, 0);
    pi.insert(2, 0) = R(2, 0);
    pi.insert(3, 0) = R(0, 1);
    pi.insert(4, 0) = R(1, 1);
    pi.insert(5, 0) = R(2, 1);
    pi.insert(6, 0) = R(0, 2);
    pi.insert(7, 0) = R(1, 2);
    pi.insert(8, 0) = R(2, 2);

    sparse_matrix_type const projection = weight * Ai_Si_.transpose() * Bi * pi;
    for (int k = 0; k < projection.outerSize(); ++k)
        for (Eigen::SparseMatrix<scalar_type>::InnerIterator it(projection, k); it; ++it)
            b(it.row()) += it.value();*/

    scalar_type const& p1 = R(0, 0);
    scalar_type const& p2 = R(1, 0);
    scalar_type const& p3 = R(2, 0);
    scalar_type const& p4 = R(0, 1);
    scalar_type const& p5 = R(1, 1);
    scalar_type const& p6 = R(2, 1);
    scalar_type const& p7 = R(0, 2);
    scalar_type const& p8 = R(1, 2);
    scalar_type const& p9 = R(2, 2);

    auto const& d11 = DmInv_(0, 0);
    auto const& d21 = DmInv_(1, 0);
    auto const& d31 = DmInv_(2, 0);
    auto const& d12 = DmInv_(0, 1);
    auto const& d22 = DmInv_(1, 1);
    auto const& d32 = DmInv_(2, 1);
    auto const& d13 = DmInv_(0, 2);
    auto const& d23 = DmInv_(1, 2);
    auto const& d33 = DmInv_(2, 2);

    scalar_type const _d11_d21_d31 = -d11 - d21 - d31;
    scalar_type const _d12_d22_d32 = -d12 - d22 - d32;
    scalar_type const _d13_d23_d33 = -d13 - d23 - d33;

    // we have already symbolically computed wi * (Ai*Si)^T * Bi * pi
    scalar_type const bi0 = (d11 * p1) + (d12 * p4) + (d13 * p7);
    scalar_type const bi1 = (d11 * p2) + (d12 * p5) + (d13 * p8);
    scalar_type const bi2 = (d11 * p3) + (d12 * p6) + (d13 * p9);
    scalar_type const bj0 = (d21 * p1) + (d22 * p4) + (d23 * p7);
    scalar_type const bj1 = (d21 * p2) + (d22 * p5) + (d23 * p8);
    scalar_type const bj2 = (d21 * p3) + (d22 * p6) + (d23 * p9);
    scalar_type const bk0 = (d31 * p1) + (d32 * p4) + (d33 * p7);
    scalar_type const bk1 = (d31 * p2) + (d32 * p5) + (d33 * p8);
    scalar_type const bk2 = (d31 * p3) + (d32 * p6) + (d33 * p9);
    scalar_type const bl0 = p1 * (_d11_d21_d31) + p4 * (_d12_d22_d32) + p7 * (_d13_d23_d33);
    scalar_type const bl1 = p2 * (_d11_d21_d31) + p5 * (_d12_d22_d32) + p8 * (_d13_d23_d33);
    scalar_type const bl2 = p3 * (_d11_d21_d31) + p6 * (_d12_d22_d32) + p9 * (_d13_d23_d33);

    b(vi + 0) += weight * bi0;
    b(vi + 1) += weight * bi1;
    b(vi + 2) += weight * bi2;
    b(vj + 0) += weight * bj0;
    b(vj + 1) += weight * bj1;
    b(vj + 2) += weight * bj2;
    b(vk + 0) += weight * bk0;
    b(vk + 1) += weight * bk1;
    b(vk + 2) += weight * bk2;
    b(vl + 0) += weight * bl0;
    b(vl + 1) += weight * bl1;
    b(vl + 2) += weight * bl2;
}

std::vector<Eigen::Triplet<deformation_gradient_constraint_t::scalar_type>>
deformation_gradient_constraint_t::get_wi_SiT_AiT_Ai_Si(
    positions_type const& p,
    masses_type const& M) const
{
    auto const N  = p.rows();
    auto const v1 = this->indices().at(0);
    auto const v2 = this->indices().at(1);
    auto const v3 = this->indices().at(2);
    auto const v4 = this->indices().at(3);

    std::size_t const vi = static_cast<std::size_t>(3) * v1;
    std::size_t const vj = static_cast<std::size_t>(3) * v2;
    std::size_t const vk = static_cast<std::size_t>(3) * v3;
    std::size_t const vl = static_cast<std::size_t>(3) * v4;

    auto const w         = this->wi();
    scalar_type const V0 = std::abs(V0_);
    auto const weight    = w * V0;

    // Uncomment to use sparse matrix product to compute wi (Ai*Si)^T * (Ai*Si)
    /*sparse_matrix_type const SiT_AiT_Ai_Si = weight * Ai_Si_.transpose() * Ai_Si_;

    std::vector<Eigen::Triplet<scalar_type>> triplets;
    triplets.reserve(SiT_AiT_Ai_Si.nonZeros());

    for (int k = 0; k < SiT_AiT_Ai_Si.outerSize(); ++k)
        for (Eigen::SparseMatrix<scalar_type>::InnerIterator it(SiT_AiT_Ai_Si, k); it; ++it)
            triplets.push_back(
                {static_cast<int>(it.row()), static_cast<int>(it.col()), it.value()});

    return triplets;*/

    // We symbolically precomputed the product (Ai*Si)^T * (Ai*Si), 
    // so here, we directly compute the non-zero entries of wi * (Ai*Si)^T * (Ai*Si)
    // without performing sparse matrix products for optimization.
    auto const& d11 = DmInv_(0, 0);
    auto const& d21 = DmInv_(1, 0);
    auto const& d31 = DmInv_(2, 0);
    auto const& d12 = DmInv_(0, 1);
    auto const& d22 = DmInv_(1, 1);
    auto const& d32 = DmInv_(2, 1);
    auto const& d13 = DmInv_(0, 2);
    auto const& d23 = DmInv_(1, 2);
    auto const& d33 = DmInv_(2, 2);

    // precompute often used quantities
    scalar_type const _d11_d21_d31 = -d11 - d21 - d31;
    scalar_type const _d12_d22_d32 = -d12 - d22 - d32;
    scalar_type const _d13_d23_d33 = -d13 - d23 - d33;

    // col 1
    scalar_type const s1_1  = d11 * d11 + d12 * d12 + d13 * d13;
    scalar_type const s4_1  = d11 * d21 + d12 * d22 + d13 * d23;
    scalar_type const s7_1  = d11 * d31 + d12 * d32 + d13 * d33;
    scalar_type const s10_1 = d11 * _d11_d21_d31 + d12 * _d12_d22_d32 + d13 * _d13_d23_d33;
    // col 2
    scalar_type const s2_2  = s1_1;
    scalar_type const s5_2  = s4_1;
    scalar_type const s8_2  = s7_1;
    scalar_type const s11_2 = s10_1;
    // col 3
    scalar_type const s3_3  = s1_1;
    scalar_type const s6_3  = s4_1;
    scalar_type const s9_3  = s7_1;
    scalar_type const s12_3 = s10_1;
    // col 4
    scalar_type const s1_4  = d11 * d21 + d12 * d22 + d13 * d23;
    scalar_type const s4_4  = d21 * d21 + d22 * d22 + d23 * d23;
    scalar_type const s7_4  = d31 * d21 + d32 * d22 + d33 * d23;
    scalar_type const s10_4 = d21 * (_d11_d21_d31) + d22 * (_d12_d22_d32) + d23 * (_d13_d23_d33);
    // col 5
    scalar_type const s2_5  = s1_4;
    scalar_type const s5_5  = s4_4;
    scalar_type const s8_5  = s7_4;
    scalar_type const s11_5 = s10_4;
    // col 6
    scalar_type const s3_6  = s1_4;
    scalar_type const s6_6  = s4_4;
    scalar_type const s9_6  = s7_4;
    scalar_type const s12_6 = s10_4;

    // col 7
    scalar_type const s1_7  = d11 * d31 + d12 * d32 + d13 * d33;
    scalar_type const s4_7  = d21 * d31 + d22 * d32 + d23 * d33;
    scalar_type const s7_7  = d31 * d31 + d32 * d32 + d33 * d33;
    scalar_type const s10_7 = _d11_d21_d31 * d31 + _d12_d22_d32 * d32 + _d13_d23_d33 * d33;
    // col 8
    scalar_type const s2_8  = s1_7;
    scalar_type const s5_8  = s4_7;
    scalar_type const s8_8  = s7_7;
    scalar_type const s11_8 = s10_7;
    // col 9
    scalar_type const s3_9  = s1_7;
    scalar_type const s6_9  = s4_7;
    scalar_type const s9_9  = s7_7;
    scalar_type const s12_9 = s10_7;
    // col 10
    scalar_type const s1_10  = _d11_d21_d31 * d11 + _d12_d22_d32 * d12 + _d13_d23_d33 * d13;
    scalar_type const s4_10  = _d11_d21_d31 * d21 + _d12_d22_d32 * d22 + _d13_d23_d33 * d23;
    scalar_type const s7_10  = _d11_d21_d31 * d31 + _d12_d22_d32 * d32 + _d13_d23_d33 * d33;
    scalar_type const s10_10 = (_d11_d21_d31) * (_d11_d21_d31) + (_d12_d22_d32) * (_d12_d22_d32) +
                               (_d13_d23_d33) * (_d13_d23_d33);
    // col 11
    scalar_type const s2_11  = s1_10;
    scalar_type const s5_11  = s4_10;
    scalar_type const s8_11  = s7_10;
    scalar_type const s11_11 = s10_10;

    // col 12
    scalar_type const s3_12  = s1_10;
    scalar_type const s6_12  = s4_10;
    scalar_type const s9_12  = s7_10;
    scalar_type const s12_12 = s10_10;

    std::array<Eigen::Triplet<scalar_type>, 12u * 4u> triplets;
    int const row1  = vi;
    int const row2  = vi + 1;
    int const row3  = vi + 2;
    int const row4  = vj;
    int const row5  = vj + 1;
    int const row6  = vj + 2;
    int const row7  = vk;
    int const row8  = vk + 1;
    int const row9  = vk + 2;
    int const row10 = vl;
    int const row11 = vl + 1;
    int const row12 = vl + 2;

    int const col1  = row1;
    int const col2  = row2;
    int const col3  = row3;
    int const col4  = row4;
    int const col5  = row5;
    int const col6  = row6;
    int const col7  = row7;
    int const col8  = row8;
    int const col9  = row9;
    int const col10 = row10;
    int const col11 = row11;
    int const col12 = row12;
    // col 1
    triplets[0] = {row1, col1, weight * s1_1};
    triplets[1] = {row4, col1, weight * s4_1};
    triplets[2] = {row7, col1, weight * s7_1};
    triplets[3] = {row10, col1, weight * s10_1};
    // col 2
    triplets[4] = {row2, col2, weight * s2_2};
    triplets[5] = {row5, col2, weight * s5_2};
    triplets[6] = {row8, col2, weight * s8_2};
    triplets[7] = {row11, col2, weight * s11_2};
    // col 3
    triplets[8]  = {row3, col3, weight * s3_3};
    triplets[9]  = {row6, col3, weight * s6_3};
    triplets[10] = {row9, col3, weight * s9_3};
    triplets[11] = {row12, col3, weight * s12_3};
    // col 4
    triplets[12] = {row1, col4, weight * s1_4};
    triplets[13] = {row4, col4, weight * s4_4};
    triplets[14] = {row7, col4, weight * s7_4};
    triplets[15] = {row10, col4, weight * s10_4};
    // col 5
    triplets[16] = {row2, col5, weight * s2_5};
    triplets[17] = {row5, col5, weight * s5_5};
    triplets[18] = {row8, col5, weight * s8_5};
    triplets[19] = {row11, col5, weight * s11_5};
    // col 6
    triplets[20] = {row3, col6, weight * s3_6};
    triplets[21] = {row6, col6, weight * s6_6};
    triplets[22] = {row9, col6, weight * s9_6};
    triplets[23] = {row12, col6, weight * s12_6};
    // col 7
    triplets[24] = {row1, col7, weight * s1_7};
    triplets[25] = {row4, col7, weight * s4_7};
    triplets[26] = {row7, col7, weight * s7_7};
    triplets[27] = {row10, col7, weight * s10_7};
    // col 8
    triplets[28] = {row2, col8, weight * s2_8};
    triplets[29] = {row5, col8, weight * s5_8};
    triplets[30] = {row8, col8, weight * s8_8};
    triplets[31] = {row11, col8, weight * s11_8};
    // col 9
    triplets[32] = {row3, col9, weight * s3_9};
    triplets[33] = {row6, col9, weight * s6_9};
    triplets[34] = {row9, col9, weight * s9_9};
    triplets[35] = {row12, col9, weight * s12_9};
    // col 10
    triplets[36] = {row1, col10, weight * s1_10};
    triplets[37] = {row4, col10, weight * s4_10};
    triplets[38] = {row7, col10, weight * s7_10};
    triplets[39] = {row10, col10, weight * s10_10};
    // col 11
    triplets[40] = {row2, col11, weight * s2_11};
    triplets[41] = {row5, col11, weight * s5_11};
    triplets[42] = {row8, col11, weight * s8_11};
    triplets[43] = {row11, col11, weight * s11_11};
    // col 12
    triplets[44] = {row3, col12, weight * s3_12};
    triplets[45] = {row6, col12, weight * s6_12};
    triplets[46] = {row9, col12, weight * s9_12};
    triplets[47] = {row12, col12, weight * s12_12};

    return std::vector<Eigen::Triplet<scalar_type>>{triplets.begin(), triplets.end()};
}

} // namespace pd
