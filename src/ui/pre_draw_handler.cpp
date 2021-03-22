#include "ui/pre_draw_handler.h"

namespace ui {

bool pre_draw_handler_t::operator()(igl::opengl::glfw::Viewer& viewer)
{
    pd::deformable_mesh_t* model = solver->model();

    if (!is_model_ready())
        return false;

    for (auto i = 0; i < model->mass().rows(); ++i)
    {
        if (model->is_fixed(i))
            continue;

        auto const eq = [](double const a, double const b) {
            double constexpr eps = 1e-5;
            double const diff    = std::abs(a - b);
            return diff <= eps;
        };

        if (!eq(model->mass()(i), static_cast<double>(physics_params->mass_per_particle)))
        {
            model->mass()(i) = static_cast<double>(physics_params->mass_per_particle);
            solver->set_dirty();
        }
    }

    if (viewer.core().is_animating)
    {
        fext->col(1).array() -= physics_params->is_gravity_active ? 9.81 : 0.;

        if (!solver->ready())
        {
            solver->prepare(physics_params->dt);
        }

        solver->step(*fext, physics_params->solver_iterations);

        fext->setZero();
        viewer.data().clear();
        viewer.data().set_mesh(model->positions(), model->faces());
    }

    for (auto i = 0u; i < model->positions().rows(); ++i)
    {
        if (!model->is_fixed(i))
            continue;

        viewer.data().add_points(model->positions().row(i), Eigen::RowVector3d{1., 0., 0.});
    }

    return false; // do not return from drawing loop
}

} // namespace ui
