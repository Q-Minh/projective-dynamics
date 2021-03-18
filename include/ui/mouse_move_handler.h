#ifndef PD_UI_MOUSE_MOVE_HANDLER_H
#define PD_UI_MOUSE_MOVE_HANDLER_H

#include "pd/deformable_mesh.h"
#include "picking_state.h"

#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject.h>

namespace ui {

struct mouse_move_handler_t
{
    std::function<bool()> is_model_ready;
    picking_state_t* picking_state;
    pd::deformable_mesh_t* model;
    Eigen::MatrixX3d* fext;

    mouse_move_handler_t(
        std::function<bool()> is_model_ready,
        picking_state_t* picking_state,
        pd::deformable_mesh_t* model,
        Eigen::MatrixX3d* fext)
        : is_model_ready(is_model_ready), picking_state(picking_state), model(model), fext(fext)
    {
    }

    bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
};

} // namespace ui

#endif // PD_UI_MOUSE_MOVE_HANDLER_H