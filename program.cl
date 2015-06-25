__kernel void calculate(__global int* r, __global int* g, __global int* b, __global int* pan, __global int* wr, __global int* wg, __global int* wb)
{
    unsigned int idx = get_global_id(0);
  
    float k = 0.5f;
    float i = (r[idx] * 0.171f + g[idx] * 0.2f + b[idx] * 0.171f) / 0.632f;
    float kx__pan_minus_iii = k * ((float) pan[idx] - i);
    float coe = (i + kx__pan_minus_iii)?(float) pan[idx] / (i + kx__pan_minus_iii):0;
    wr[idx] = coe * (r[idx] + kx__pan_minus_iii);
    wg[idx] = coe * (g[idx] + kx__pan_minus_iii);
    wb[idx] = coe * (b[idx] + kx__pan_minus_iii);
}
