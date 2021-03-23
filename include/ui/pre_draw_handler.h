#ifndef PD_UI_PRE_DRAW_HANDLER_H
#define PD_UI_PRE_DRAW_HANDLER_H

#include "pd/solver.h"
#include "ui/physics_params.h"

#include <igl/opengl/glfw/Viewer.h>

namespace ui {

struct pre_draw_handler_t
{
    std::function<bool()> is_model_ready;
    physics_params_t* physics_params;
    pd::solver_t* solver;
    Eigen::MatrixX3d* fext;

    pre_draw_handler_t(
        std::function<bool()> is_model_ready,
        physics_params_t* physics_params,
        pd::solver_t* solver,
        Eigen::MatrixX3d* fext)
        : is_model_ready(is_model_ready), physics_params(physics_params), solver(solver), fext(fext)
    {
    }

    bool operator()(igl::opengl::glfw::Viewer& viewer);
};

} // namespace ui

#endif // PD_UI_PRE_DRAW_HANDLER_H