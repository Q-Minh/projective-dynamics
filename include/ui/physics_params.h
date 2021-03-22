#ifndef PD_UI_PHYSICS_PARAMS_H
#define PD_UI_PHYSICS_PARAMS_H

namespace ui {

struct physics_params_t
{
    bool is_gravity_active  = false;
    float dt                = 0.0166667;
    int solver_iterations   = 10;
    int solver_substeps     = 10;
    float mass_per_particle = 1000.f;
    float alpha             = 0.00000001f;
    float wi                = 1000.f;

    // fem
    float young_modulus = 10000.f;
    float poisson_ratio = 0.49f;
};

} // namespace ui

#endif // PD_UI_PHYSICS_PARAMS_H