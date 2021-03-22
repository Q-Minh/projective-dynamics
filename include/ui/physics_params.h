#ifndef PD_UI_PHYSICS_PARAMS_H
#define PD_UI_PHYSICS_PARAMS_H

namespace ui {

struct physics_params_t
{
    bool is_gravity_active         = false;
    float dt                       = 0.0166667;
    int solver_iterations          = 10;
    float mass_per_particle        = 1000.f;
    float edge_constraint_wi       = 1000.f;
    float positional_constraint_wi = 100'000.f;

    // fem
    float young_modulus = 10000.f;
    float poisson_ratio = 0.49f;
};

} // namespace ui

#endif // PD_UI_PHYSICS_PARAMS_H