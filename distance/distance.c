#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "alvincl/alvincl.h"

#define MEM_ALIGNMENT			256

#define DEVICE_TYPE				CL_DEVICE_TYPE_GPU

#define ARRAY_SIZE_A			2048
#define ARRAY_SIZE_B			40000
#define ARRAY_SIZE_RESULT		81920000

typedef float dtype;
typedef cl_float2 dtype2;

inline dtype distance(dtype2 a, dtype2 b){
    return sqrtf(powf((a.x - b.x), 2) + powf((a.y - b.y), 2));
}

//~ inline dtype distance(dtype2 a, dtype2 b){
//~ 
//~ a.x -= b.x;
//~ a.y -= b.y;
//~ a.y *= a.y;
//~ a.x = a.x * a.x + a.y;
//~ a.x = sqrtf(a.x);
//~ 
//~ return a.x;
//~ }

cl_ulong timeNanos(){
#ifdef linux
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (unsigned long long) tp.tv_sec * (1000ULL * 1000ULL * 1000ULL) + (unsigned long long) tp.tv_nsec;
#else
    LARGE_INTEGER current;
    QueryPerformanceCounter(&current);
    return (unsigned long long)((double)current.QuadPart / m_ticksPerSec * 1e9);
#endif
}


int main() {
    cl_uint num_platforms;
    platform_struct *platforms = NULL;
    cl_uint2 ddex;

    size_t global_work_group_size[2] = { 2048, 40000 };
    size_t local_work_group_size[2] = { 32, 2 };

    cl_event kernel_event;
    cl_event a_map_event, b_map_event, rst_map_event;
    cl_event a_unmap_event, b_unmap_event, rst_unmap_event;

    cl_ulong use, start, end;

    cl_int ret_num;

    int i, j;

    dtype2* a = NULL;
    dtype2* b = NULL;
    dtype* result = NULL;;

    // Alloc aligned a, b
    ret_num = posix_memalign((void**)&a, MEM_ALIGNMENT, ARRAY_SIZE_A * sizeof(dtype2));
    checkPointer(platforms, a, "a");

    ret_num = posix_memalign((void**)&b, MEM_ALIGNMENT, ARRAY_SIZE_B * sizeof(dtype2));
    checkPointer(platforms, b, "b");


    // Init a, b
    for (i = 0; i < ARRAY_SIZE_A; i++)
    {
        a[i].x = (dtype)(64.0*rand()/(RAND_MAX+1.0));
        a[i].y = (dtype)(64.0*rand()/(RAND_MAX+1.0));
    }
    for (i = 0; i < ARRAY_SIZE_B; i++)
    {
        b[i].x = (dtype)(64.0*rand()/(RAND_MAX+1.0));
        b[i].y = (dtype)(64.0*rand()/(RAND_MAX+1.0));
    }

    //
    platforms = getPlatforms(&num_platforms);
    getDevices(platforms);	
    ddex = setDevice(platforms, DEVICE_TYPE);	
    printf("select device %s \n", platforms[ddex.x].devices[ddex.y].name);	
    createContext(platforms, ddex);	
    createProgram(platforms, ddex, "distance.cl");	
    platforms[ddex.x].kernel = clCreateKernel(platforms[ddex.x].program, "distance_kernel", &ret_num);
    checkResult(platforms, ret_num, "clCreateKernel");		
    createCommandQueue(platforms, ddex, CL_QUEUE_PROFILING_ENABLE);
    initMemoryObjects(platforms, ddex, 3);

    platforms[ddex.x].mem_objects[0] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, ARRAY_SIZE_A * sizeof(dtype2), a, &ret_num);
    checkResult(platforms, ret_num, "clCreateBuffer(a)");

    platforms[ddex.x].mem_objects[1] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, ARRAY_SIZE_B * sizeof(dtype2), b, &ret_num);
    checkResult(platforms, ret_num, "clCreateBuffer(b)");

    platforms[ddex.x].mem_objects[2] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, ARRAY_SIZE_RESULT * sizeof(dtype), NULL, &ret_num);
    checkResult(platforms, ret_num, "clCreateBuffer(c)");

    ret_num = clSetKernelArg(platforms[ddex.x].kernel, 0, sizeof(cl_mem), &platforms[ddex.x].mem_objects[0]);
    checkResult(platforms, ret_num, "clSetKernelArg(a)");

    ret_num = clSetKernelArg(platforms[ddex.x].kernel, 1, sizeof(cl_mem), &platforms[ddex.x].mem_objects[1]);
    checkResult(platforms, ret_num, "clSetKernelArg(b)");

    ret_num = clSetKernelArg(platforms[ddex.x].kernel, 2, sizeof(cl_mem), &platforms[ddex.x].mem_objects[2]);
    checkResult(platforms, ret_num, "clSetKernelArg(c)");

    // GPU
    ret_num = clEnqueueNDRangeKernel(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].kernel, 2, NULL, global_work_group_size, local_work_group_size, 0, NULL, &kernel_event);
    checkResult(platforms, ret_num, "clEnqueueNDRangeKernel");	
    while(clWaitForEvents(1, &kernel_event) != CL_SUCCESS){

    }

    // Map MemObject	
    a = clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], CL_TRUE, CL_MAP_READ, 0, ARRAY_SIZE_A * sizeof(dtype2), 0, NULL, &a_map_event, &ret_num);
    checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects[0])");
    while(clWaitForEvents(1, &a_map_event) != CL_SUCCESS){

    }

    b = clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[1], CL_TRUE, CL_MAP_READ, 0, ARRAY_SIZE_B * sizeof(dtype2), 0, NULL, &b_map_event, &ret_num);
    checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects[1])");
    while(clWaitForEvents(1, &b_map_event) != CL_SUCCESS){

    }

    result = clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[2], CL_TRUE, CL_MAP_READ, 0, ARRAY_SIZE_RESULT * sizeof(float), 0, NULL, &rst_map_event, &ret_num);
    checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects[2])");
    while(clWaitForEvents(1, &rst_map_event) != CL_SUCCESS){

    }

    // CPU
    start = timeNanos();
    for(i = 0; i < ARRAY_SIZE_B; i++){
        for(j = 0; j < ARRAY_SIZE_A; j++){
            result[(i << 11) + j] = distance(a[j], b[i]);
        }
    }
    end = timeNanos();

    // Output
    use = end - start;
    printf("CPU Time:%3llu.%09llus (1 core)\n", (unsigned long long)use / 1000000000, (unsigned long long)use % 1000000000);


    ret_num = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(start))");
    ret_num = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(quit))");
    use = end - start;
    printf("GPU Time:%3llu.%09llus\n", (unsigned long long)use / 1000000000, (unsigned long long)use % 1000000000);

    // Unmap MemObject
    ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], a, 0, NULL, &a_unmap_event);
    checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
    while(clWaitForEvents(1, &a_unmap_event) != CL_SUCCESS){

    }

    ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[1], b, 0, NULL, &b_unmap_event);
    checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
    while(clWaitForEvents(1, &b_unmap_event) != CL_SUCCESS){

    }

    ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[2], result, 0, NULL, &rst_unmap_event);
    checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
    while(clWaitForEvents(1, &rst_unmap_event) != CL_SUCCESS){

    }

    //
    cleanUp(platforms);

    return 0;
}
