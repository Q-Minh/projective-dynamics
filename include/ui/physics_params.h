#ifndef PD_UI_PHYSICS_PARAMS_H
#define PD_UI_PHYSICS_PARAMS_H

namespace ui {

struct physics_params_t
{
    bool is_gravity_active                   = false;
    float dt                                 = 0.0166667;
    int solver_iterations                    = 10;
    float mass_per_particle                  = 10.f;
    float edge_constraint_wi                 = 1'000'000.f;
    float positional_constraint_wi           = 1'000'000'000.f;
    float deformation_gradient_constraint_wi = 10'000'000.f;
    float strain_limit_constraint_wi         = 10'000'000.f;
};

} // namespace ui

#endif // PD_UI_PHYSICS_PARAMS_H