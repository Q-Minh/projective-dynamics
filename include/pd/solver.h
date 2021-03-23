#ifndef PD_PD_SIMULATION_H
#define PD_PD_SIMULATION_H

#include "deformable_mesh.h"

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <algorithm>
#include <iostream>
#include <vector>

namespace pd {
namespace detail {

inline Eigen::VectorXd flatten(Eigen::MatrixXd const& p)
{
    auto const N = p.rows();
    Eigen::VectorXd q;
    q.resize(3 * N);
    for (std::size_t i = 0; i < N; ++i)
        q.block(3u * i, 0, 3, 1) = p.row(i).transpose();

    return q;
}

inline Eigen::MatrixXd unflatten(Eigen::VectorXd const& q)
{
    auto const N = q.rows() / 3;
    Eigen::MatrixXd p(N, 3);
    for (std::size_t i = 0; i < N; ++i)
        p.row(i) = q.block(3 * i, 0, 3, 1).transpose();

    return p;
}

} // namespace detail

class solver_t
{
  public:
    using scalar_type = typename deformable_mesh_t::scalar_type;

    void set_model(deformable_mesh_t* model)
    {
        model_ = model;
        set_dirty();
    }
    deformable_mesh_t const* model() const { return model_; }
    deformable_mesh_t* model() { return model_; }
    void set_dirty() { dirty_ = true; }
    void set_clean() { dirty_ = false; }
    bool ready() const { return !dirty_; }
    void prepare(scalar_type dt)
    {
        dt_                   = dt;
        auto const& positions = model_->positions();
        auto const& mass      = model_->mass();
        auto const N          = positions.rows();

        auto const dt2_inv = scalar_type{1.} / (dt * dt);

        std::vector<Eigen::Triplet<scalar_type>> A_triplets;
        // vectors double their size on each reallocation.
        // we decrease the number of reallocations by preallocating a large chunk
        // upfront to improve performance. 3*N is just a good initial starting capacity
        A_triplets.reserve(3 * N);
        auto& constraints = model_->constraints();
        for (auto& constraint : constraints)
        {
            auto const SiT_AiT_Ai_Si = constraint->get_wi_SiT_AiT_Ai_Si(positions, mass);
            A_triplets.insert(A_triplets.end(), SiT_AiT_Ai_Si.begin(), SiT_AiT_Ai_Si.end());
        }

        for (auto i = 0; i < N; ++i)
        {
            A_triplets.push_back({3 * i + 0, 3 * i + 0, mass(i) * dt2_inv});
            A_triplets.push_back({3 * i + 1, 3 * i + 1, mass(i) * dt2_inv});
            A_triplets.push_back({3 * i + 2, 3 * i + 2, mass(i) * dt2_inv});
        }

        Eigen::SparseMatrix<scalar_type> A(3 * N, 3 * N);
        A.setFromTriplets(A_triplets.begin(), A_triplets.end());

        cholesky_decomposition_.compute(A);

        set_clean();
    }

    void step(Eigen::MatrixXd const& fext, int num_iterations = 10)
    {
        auto const& constraints = model_->constraints();
        auto& positions         = model_->positions();
        auto& velocities        = model_->velocity();
        auto const& mass        = model_->mass();
        auto const N            = positions.rows();

        auto const dt      = dt_;
        auto const dt_inv  = scalar_type{1.} / dt_;
        auto const dt2     = dt_ * dt_;
        auto const dt2_inv = scalar_type{1.} / dt2;
        // q_explicit = q(t) + dt*v(t) + dt^2 * M^(-1) * fext(t)
        Eigen::MatrixX3d const a                   = fext.array().colwise() / mass.array();
        Eigen::MatrixXd const explicit_integration = positions + dt * velocities + dt2 * a;

        // sn = flatten(q_explicit)
        // format of sn is [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn]^T
        Eigen::VectorXd sn = detail::flatten(explicit_integration);

        // the matrix-vector product: (M / dt^2) * sn
        Eigen::VectorXd masses;
        masses.resize(3 * N);
        for (std::size_t i = 0; i < N; ++i)
        {
            Eigen::Matrix3d M;
            M.setZero();
            M(0, 0) = mass(i);
            M(1, 1) = mass(i);
            M(2, 2) = mass(i);

            auto const sni               = sn.block(3u * i, 0, 3, 1);
            masses.block(3 * i, 0, 3, 1) = dt2_inv * M * sni;
        }

        // initial q(t+1)
        Eigen::VectorXd q = sn;

        Eigen::VectorXd b;
        b.resize(3 * N);

        for (int k = 0; k < num_iterations; ++k)
        {
            // b = (M/dt^2)*sn + sum wi * (Ai*Si)^T * (Ai*Si)
            b.setZero();
            for (auto const& constraint : constraints)
            {
                constraint->project_wi_SiT_AiT_Bi_pi(q, b);
            }
            b += masses;

            // Ax = b
            q = cholesky_decomposition_.solve(b);
        }

        Eigen::MatrixXd const qn_plus_1 = detail::unflatten(q);
        velocities                      = (qn_plus_1 - positions) * dt_inv;
        positions                       = qn_plus_1;
    }

  private:
    deformable_mesh_t* model_;
    bool dirty_;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<scalar_type>> cholesky_decomposition_;
    Eigen::MatrixXd A_;
    scalar_type dt_;
};

} // namespace pd

#endif // PD_PD_SIMULATION_H