__kernel void collision(
    __global float* f,    // Distribution functions (flattened: (N, Q) -> 1D array)
    __global float* f_temp,  // Temporary buffer
    __global float* u,    // Velocity field
    __global float* rho,  // Density field
    const float tau,      // Relaxation time
    const int Nx,         // Number of lattice nodes in x-direction
    const int Ny          // Number of lattice nodes in y-direction
) {
    int idx = get_global_id(0);  // Node index (flattened 1D)

    // Constants for equilibrium calculations
    const float w[9] = {4.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};
    const int c[9][2] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1},
                         {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

    float omega = 1.0f / tau;

    // Extract local variables
    float local_density = rho[idx];
    float ux = u[2 * idx];
    float uy = u[2 * idx + 1];

    float u_dot_u = ux * ux + uy * uy;
    float f_eq[9];

    // Calculate equilibrium distribution
    for (int q = 0; q < 9; q++) {
        float ci_dot_u = c[q][0] * ux + c[q][1] * uy;
        f_eq[q] = w[q] * local_density * (1 + 3 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * u_dot_u);
    }

    // Perform collision step
    for (int q = 0; q < 9; q++) {
        f[idx * 9 + q] = f[idx * 9 + q] + omega * (f_eq[q] - f[idx * 9 + q]);
    }
}

__kernel void streaming(
    __global float* f,    // Distribution functions (flattened: (N, Q) -> 1D array)
    __global float* f_temp,  // Temporary buffer
    __global float* u,    // Velocity field
    __global float* rho,  // Density field
    const float tau,      // Relaxation time
    const int Nx,         // Number of lattice nodes in x-direction
    const int Ny          // Number of lattice nodes in y-direction
) {
    int idx = get_global_id(0);  // Node index (flattened 1D)
    int x = idx % Nx;
    int y = idx / Nx;

    const int c[9][2] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1},
                         {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

    // Streaming step
    for (int q = 0; q < 9; q++) {
        int x_new = (x + c[q][0] + Nx) % Nx;
        int y_new = (y + c[q][1] + Ny) % Ny;
        int idx_new = y_new * Nx + x_new;

        f[idx_new * 9 + q] = f[idx * 9 + q];
    }
}
