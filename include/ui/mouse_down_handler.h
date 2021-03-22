#ifndef PD_UI_MOUSE_DOWN_HANDLER_H
#define PD_UI_MOUSE_DOWN_HANDLER_H

#include "pd/deformable_mesh.h"
#include "pd/solver.h"
#include "physics_params.h"
#include "picking_state.h"

#include <GLFW/glfw3.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>

namespace ui {

struct mouse_down_handler_t
{
    std::function<bool()> is_model_ready;
    picking_state_t* picking_state;
    pd::solver_t* solver;
    physics_params_t* physics_params;

    mouse_down_handler_t(
        std::function<bool()> is_model_ready,
        picking_state_t* picking_state,
        pd::solver_t* solver,
        physics_params_t* physics_params)
        : is_model_ready(is_model_ready),
          picking_state(picking_state),
          solver(solver),
          physics_params(physics_params)
    {
    }

    bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
};

} // namespace ui

#endif // PD_UI_MOUSE_DOWN_HANDLER_H