#include <chrono>
#include <geometry/get_simple_bar_model.h>
#include <matplot/matplot.h>
#include <pd/deformable_mesh.h>
#include <pd/solver.h>

int main(int argc, char** argv)
{
    auto const rescale = [](Eigen::MatrixXd& V) {
        Eigen::RowVector3d v_mean = V.colwise().mean();
        V.rowwise() -= v_mean;
        V.array() /= V.maxCoeff() - V.minCoeff();
    };

    auto const compute_total_strain = [](pd::deformable_mesh_t const& mesh) {
        double const total_strain = std::accumulate(
            mesh.constraints().begin(),
            mesh.constraints().end(),
            0.,
            [&](double const sum, std::unique_ptr<pd::constraint_t> const& constraint) {
                double const C = constraint->evaluate(mesh.positions(), mesh.mass());
                return sum + C;
            });

        return total_strain;
    };

    std::vector<double> y{};
    std::vector<int> x{};
    for (int num_iterations = 0; num_iterations < 10; ++num_iterations)
    {
        std::size_t const width = 12u, height = 4u, depth = 4u;
        auto [V, T, F] = geometry::get_simple_bar_model(width, height, depth);
        rescale(V);
        Eigen::VectorXd masses(V.rows());
        masses.setConstant(10.);
        pd::deformable_mesh_t mesh{V, F, T, masses};

        mesh.velocity().rowwise() += Eigen::RowVector3d{5., -5., 0.};

        double const deformation_gradient_wi = 10'000'000.;
        double const positional_wi           = 1'000'000'000.;
        mesh.constrain_deformation_gradient(deformation_gradient_wi);
        auto const num_fixed_particles = height * depth;
        for (std::size_t i = 0u; i < num_fixed_particles; ++i)
        {
            mesh.add_positional_constraint(i, positional_wi);
            mesh.fix(i);
        }

        Eigen::MatrixXd fext(V.rows(), 3u);
        fext.setZero();
        fext.col(1).array() -= 9.81;

        auto const offset_from_last_bar_slice = V.rows() - (height * depth);
        auto force_affected_block = fext.block(offset_from_last_bar_slice, 0u, height * depth, 3u);
        double const force        = 10'000.; // 10 kN
        force_affected_block.rowwise() += Eigen::RowVector3d{0., -force, 0.};

        pd::solver_t solver{};
        solver.set_model(&mesh);
        double const dt = 0.03333333333333333;
        if (!solver.ready())
        {
            solver.prepare(dt);
        }
        solver.step(fext, num_iterations);

        double const total_strain = compute_total_strain(mesh);
        y.push_back(total_strain);
        x.push_back(num_iterations);
    }

    auto fig1  = matplot::figure();
    auto axes1 = fig1->current_axes();
    fig1->title("Elastic potential w.r.t. iteration count (dt=0.033)");
    axes1->xlabel("Number of solver iterations");
    axes1->ylabel("Total strain");
    axes1->grid(true);
    axes1->plot(x, y);

    std::vector<double> y1{};
    std::vector<double> y2{};
    std::vector<int> x1{};
    std::vector<int> x2{};
    for (std::size_t i = 0u; i < 20u; ++i)
    {
        std::size_t const width = 12u + i, height = 4u + i, depth = 4u + i;
        auto [V, T, F] = geometry::get_simple_bar_model(width, height, depth);
        rescale(V);
        Eigen::VectorXd masses(V.rows());
        masses.setConstant(10.);
        pd::deformable_mesh_t mesh{V, F, T, masses};
        double const deformation_gradient_wi = 10'000'000.;
        mesh.constrain_deformation_gradient(deformation_gradient_wi);

        Eigen::MatrixXd fext(V.rows(), 3u);
        fext.setZero();
        fext.col(1).array() -= 9.81;
        double const force = 5000.; // 5 kN
        fext.row(fext.rows() - 1)(1) += -force;

        pd::solver_t solver{};
        solver.set_model(&mesh);
        double const dt = 0.0167;
        auto before     = std::chrono::high_resolution_clock::now();
        if (!solver.ready())
        {
            solver.prepare(dt);
        }
        auto now = std::chrono::high_resolution_clock::now();

        auto const prefactorization_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(now - before).count();

        int num_iterations = 10;
        before             = std::chrono::high_resolution_clock::now();
        solver.step(fext, num_iterations);
        now = std::chrono::high_resolution_clock::now();
        auto const solver_total_time =
            std::chrono::duration_cast<std::chrono::microseconds>(now - before).count();

        double const time_for_prefactorization = static_cast<double>(prefactorization_duration);
        double const average_time_per_iteration =
            static_cast<double>(solver_total_time) / static_cast<double>(num_iterations);
        int num_vertices = V.rows();

        y1.push_back(time_for_prefactorization);
        y2.push_back(average_time_per_iteration);
        x1.push_back(num_vertices);
        x2.push_back(mesh.constraints().size());
    }

    auto fig2  = matplot::figure();
    auto axes2 = fig2->current_axes();
    fig2->title("Cholesky Prefactorization Time w.r.t. Vertex Count");
    axes2->ylabel("Time (us)");
    axes2->xlabel("Number of vertices");
    axes2->grid(true);
    axes2->plot(x1, y1);

    auto fig3  = matplot::figure();
    auto axes3 = fig3->current_axes();
    fig3->title("Cholesky Prefactorization Time w.r.t. Constraint Count");
    axes3->xlabel("Number of constraints");
    axes3->ylabel("Time (us)");
    axes3->grid(true);
    axes3->plot(x2, y1);

    auto fig4  = matplot::figure();
    auto axes4 = fig4->current_axes();
    fig4->title("Average Time per Iteration w.r.t. Vertex Count");
    axes4->xlabel("Number of vertices");
    axes4->ylabel("Average Time (us)");
    axes4->grid(true);
    axes4->plot(x1, y2);

    auto fig5  = matplot::figure();
    auto axes5 = fig5->current_axes();
    fig5->title("Average Time per Iteration w.r.t. Constraint Count");
    axes5->xlabel("Number of constraints");
    axes5->ylabel("Average Time (us)");
    axes5->grid(true);
    axes5->plot(x2, y2);

    matplot::show();

    return 0;
}