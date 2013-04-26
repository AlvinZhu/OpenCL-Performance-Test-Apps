__kernel void distance_kernel(__global const float2 *a,
						__global const float2 *b,
						__global float *result)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    result[(j << 11) + i] = distance(a[i], b[j]);
    
    //~ int rindex;
    //~ float2 tmp, ta, tb;
    //~ 
    //~ rindex = (j << 11) + i;
    //~ ta = a[i];
    //~ tb = b[j];
    //~ tmp = ta - tb;
    //~ tmp.y *= tmp.y;
    //~ tmp.x = tmp.x * tmp.x + tmp.y;
    //~ tmp.x = native_sqrt(tmp.x);
    //~ result[rindex] = tmp.x;

}
