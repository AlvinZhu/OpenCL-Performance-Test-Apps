#include <stdio.h>
#include <stdlib.h>

#include "alvincl/alvincl.h"

#define DEVICE_TYPE		CL_DEVICE_TYPE_GPU
#define MAX_ARRAY_SIZE  16777216
#define NUM_ARRAY	2	// 1, 2, 3 or 4
#define MAX_LOCAL_WORK_GROUP_SIZE       128	

int main() {
	cl_uint num_platforms;
	platform_struct *platforms = NULL;
	cl_uint2 ddex;
	
	cl_int ret_num;	
	
	cl_char *buf[NUM_ARRAY];
	cl_uint array_size;
	
	cl_ulong result[8][8];
	
	cl_uint x, y, rx, ry;
	
	cl_uint i;
	
	cl_event kernel_event, write_event[NUM_ARRAY];
	
	cl_ulong queued, quit;
	cl_ulong sumbit, start;
	
	size_t globalWorkSize[2] = { 4096, 4096 };
    size_t localWorkSize[2] = { 0, 0 };
	
	platforms = getPlatforms(&num_platforms);
	getDevices(platforms);
	
	ddex = setDevice(platforms, DEVICE_TYPE);
	
	printf("select device %s \n", platforms[ddex.x].devices[ddex.y].name);
	
	createContext(platforms, ddex);
	
	createProgram(platforms, ddex, "work_group_size_test.cl");
	
	char kernelname[1024];
	sprintf(kernelname, "work_group_size_test_kernel_%d", NUM_ARRAY);
	platforms[ddex.x].kernel = clCreateKernel(platforms[ddex.x].program, kernelname, &ret_num);
	checkResult(platforms, ret_num, "clCreateKernel");
		
	createCommandQueue(platforms, ddex, CL_QUEUE_PROFILING_ENABLE);

	initMemoryObjects(platforms, ddex, NUM_ARRAY);
	
	array_size = MAX_ARRAY_SIZE;
	rx = -1;
	
	for(x = 1; x <= MAX_LOCAL_WORK_GROUP_SIZE; x *= 2){
		localWorkSize[0] = x;
		rx++;
		ry = -1;
		for(y = 1; y <= MAX_LOCAL_WORK_GROUP_SIZE; y *= 2){
			ry++;
			if( x * y > MAX_LOCAL_WORK_GROUP_SIZE){
				result[rx][ry] = 0;
				continue;
			}
			
			for(i = 0; i < NUM_ARRAY; i++){	
				buf[i] = (cl_char *) calloc(sizeof(cl_char), array_size);
			}
			
			for(i = 0; i < NUM_ARRAY; i++){			
				platforms[ddex.x].mem_objects[i] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_READ_WRITE, sizeof(char) * array_size, NULL, &ret_num);
				checkResult(platforms, ret_num, "clCreateBuffer(mem_objects)");		
			}
			
			for(i = 0; i < NUM_ARRAY; i++){
				ret_num = clEnqueueWriteBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], CL_TRUE, 0, sizeof(char) * array_size, buf[i], 0, NULL, &write_event[i]);
				checkResult(platforms, ret_num, "clEnqueueWriteBuffer(mem_objects)");
				while(clWaitForEvents(1, &write_event[i]) != CL_SUCCESS){
				
				}
			}
			
			for(i = 0; i < NUM_ARRAY; i++){
				ret_num = clSetKernelArg(platforms[ddex.x].kernel, i, sizeof(cl_mem), &platforms[ddex.x].mem_objects[i]);
				checkResult(platforms, ret_num, "clEnqueueWriteBuffer(mem_objects)");
			}
			
			localWorkSize[1] = y;

			ret_num = clEnqueueNDRangeKernel(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernel_event);
			checkResult(platforms, ret_num, "clEnqueueNDRangeKernel");
			
			while(clWaitForEvents(1, &kernel_event) != CL_SUCCESS){
				
			}
					
			printf("================================\n");
			ret_num = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(queued))");
			ret_num = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &sumbit, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(sumbit))");
			ret_num = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(start))");
			ret_num = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &quit, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(quit))");
			
			printf("kernel(COPY): %d byte\n", array_size);
			printf("localWorkSize: %dx%d \n", x, y);
			printf("         __________s__m__u__n\n");
			printf("  queued:%20.0llu ns.\n", queued);
			printf("  sumbit:%20.0llu ns.\n", sumbit);
			printf(" que-sum:%20.0llu ns.\n", sumbit - queued);
			printf("   start:%20.0llu ns.\n", start);
			printf(" sum-sta:%20.0llu ns.\n", start - sumbit);
			printf("    quit:%20.0llu ns.\n", quit);
			printf(" sta-qui:%20.0llu ns.\n", quit - start);
			printf("   total:%20.0llu ns.\n", quit - queued);
			
			result[rx][ry] = quit - queued;
			
			for(i = 0; i < NUM_ARRAY; i++){
				free((void*)buf[i]);
				ret_num = clReleaseMemObject(platforms[ddex.x].mem_objects[i]);
				checkResult(platforms, ret_num, "clReleaseMemObject(mem_objects)");
			}
		}
			
	}		
	
	for(x = 0; x < 8; x++){
		for(y = 0; y < 8; y++){
			printf("%10.1llu\t", result[x][y]);
		}
		printf("\n");
	}
	
	
	platforms[ddex.x].num_mems = 0;
	free((void*)platforms[ddex.x].mem_objects);
	
	cleanUp(platforms);
	
	return 0;
}
