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

        // Eigen::MatrixXd M(3 * N, 3 * N);
        // M.setIdentity();
        // for (auto i = 0; i < N; ++i)
        //{
        //    M.block(3 * i, 3 * i, 3, 3) *= mass(i);
        //}

        auto const dt2_inv = scalar_type{1.} / (dt * dt);
        // std::cout << "M:\n" << M << "\n";
        // M *= dt2_inv;
        // std::cout << "M / dt^2:\n" << M << "\n";

        std::vector<Eigen::Triplet<scalar_type>> A_triplets;
        // vectors double their size on each reallocation.
        // we decrease the number of reallocations by preallocating a large chunk
        // upfront to improve performance. 3*N is just a good initial starting capacity
        A_triplets.reserve(3 * N);
        // Eigen::MatrixXd A(3 * N, 3 * N);
        // A.setZero();
        auto& constraints = model_->constraints();
        for (auto& constraint : constraints)
        {
            // auto const SiT_AiT_Ai_Si = constraint->get_wi_SiT_AiT_Ai_Si_dense(positions, mass);
            auto const SiT_AiT_Ai_Si = constraint->get_wi_SiT_AiT_Ai_Si(positions, mass);
            // std::cout << "constraint:\n" << SiT_AiT_Ai_Si << "\n";
            A_triplets.insert(A_triplets.end(), SiT_AiT_Ai_Si.begin(), SiT_AiT_Ai_Si.end());
            // A += SiT_AiT_Ai_Si;
        }

        for (auto i = 0; i < N; ++i)
        {
            A_triplets.push_back({3 * i + 0, 3 * i + 0, mass(i) * dt2_inv});
            A_triplets.push_back({3 * i + 1, 3 * i + 1, mass(i) * dt2_inv});
            A_triplets.push_back({3 * i + 2, 3 * i + 2, mass(i) * dt2_inv});
        }

        Eigen::SparseMatrix<scalar_type> A(3 * N, 3 * N);
        A.setFromTriplets(A_triplets.begin(), A_triplets.end());

        // std::cout << "sum wi Si^T Ai^T Ai Si:\n" << A << "\n";
        // A += M;
        // std::cout << "M/dt^2 + sum wi Si^T Ai^T Ai Si:\n" << A << "\n";

        cholesky_decomposition_.compute(A);
        // auto const info = cholesky_decomposition_.info();

        // A_ = A;

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
        Eigen::VectorXd sn;
        sn.resize(3 * N);
        // format of sn is [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn]^T
        for (auto i = 0; i < N; ++i)
            sn.block(3 * i, 0, 3, 1) = explicit_integration.row(i).transpose();

        // std::cout << "p(t):\n" << positions << "\n";
        // std::cout << "sn(t):\n" << sn << "\n";

        // the matrix-vector product: (M / dt^2) * sn
        Eigen::VectorXd masses;
        masses.resize(3 * N);
        for (auto i = 0; i < N; ++i)
        {
            Eigen::Matrix3d M;
            M.setZero();
            M(0, 0) = mass(i);
            M(1, 1) = mass(i);
            M(2, 2) = mass(i);

            auto const sni               = sn.block(3 * i, 0, 3, 1);
            masses.block(3 * i, 0, 3, 1) = dt2_inv * M * sni;
        }

        // std::cout << "m/dt^2*sn:\n" << masses << "\n";

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
                // auto const rhs = constraint->project_wi_SiT_AiT_Bi_pi_dense(q);
                // std::cout << "constraint projection:\n" << rhs << "\n";
                // b += rhs;
            }
            // std::cout << "b sum constraints:\n" << b << "\n";
            b += masses;
            // std::cout << "b:\n" << b << "\n";

            // Ax = b
            // Eigen::VectorXd const x = cholesky_decomp.solve(b);
            q = cholesky_decomposition_.solve(b);
            // std::cout << "x:\n" << q << "\n";
        }

        for (auto i = 0; i < N; ++i)
        {
            auto const qn_plus_1 = q.block(3 * i, 0, 3, 1).transpose();
            auto& qn             = positions.row(i);
            velocities.row(i)    = (qn_plus_1 - qn) * dt_inv;
            positions.row(i)     = qn_plus_1;
        }
        // std::cout << "q(n+1):\n" << positions << "\n";
        // std::cout << "v(n+1):\n" << velocities << "\n";
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