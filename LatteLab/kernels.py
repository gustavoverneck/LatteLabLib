# kernels.py

# Imports
import cupy as cp

# Streaming kernel
stream_kernel = cp.ElementwiseKernel(
    '''
    raw float32 f,
    raw int32 c,
    raw int32 flags,
    raw int32 opp,
    int32 Q,
    int32 D,
    int32 Nx,
    int32 Ny,
    int32 Nz
    ''',
    'float32 f_new',
    '''
    // Calculate current thread's global index
    int idx = i;

    // Decompose idx into (x, y, z, q)
    int q = idx / (Nx * Ny * Nz);
    idx = idx % (Nx * Ny * Nz);
    int z = idx / (Nx * Ny);
    idx = idx % (Nx * Ny);
    int y = idx / Nx;
    int x = idx % Nx;

    // Get velocity components for direction q
    int cx = c[q * D + 0];
    int cy = c[q * D + 1];
    int cz = (D == 3) ? c[q * D + 2] : 0;

    // Compute source coordinates (periodic boundaries)
    int src_x = (x - cx + Nx) % Nx;
    int src_y = (y - cy + Ny) % Ny;
    int src_z = (D == 3) ? (z - cz + Nz) % Nz : z;

    // Check if source cell is solid
    int src_idx = src_x + src_y * Nx + src_z * Nx * Ny;
    int is_solid = flags[src_idx];

    if (is_solid) {
        // Bounce-back
        int opp_q = opp[q];
        f_new = f[x + y * Nx + z * Nx * Ny + opp_q * Nx * Ny * Nz];
    } else {
        // Regular streaming
        f_new = f[src_idx + q * Nx * Ny * Nz];
    }
    ''',
    'stream_kernel'
)