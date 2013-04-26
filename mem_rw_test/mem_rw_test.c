#include <stdio.h>
#include <stdlib.h>

#include "alvincl/alvincl.h"

#define DEVICE_TYPE		CL_DEVICE_TYPE_GPU
#define MAX_ARRAY_SIZE	16777216
#define TIMES_COPY	2

int main() {
	cl_uint num_platforms;
	platform_struct *platforms = NULL;
	cl_uint2 ddex;
	
	cl_int ret_num;	
	
	cl_char *buf;
	cl_uint array_size;
	
	cl_uint i;
	
	cl_event write_event[TIMES_COPY];
	cl_event read_event[TIMES_COPY];
	cl_event map_event[TIMES_COPY];
	cl_event unmap_event[TIMES_COPY];
	
	cl_ulong queued, sumbit, start, quit;
	
	
	
	platforms = getPlatforms(&num_platforms);
	getDevices(platforms);
	
	ddex = setDevice(platforms, DEVICE_TYPE);
	
	printf("select device %s \n", platforms[ddex.x].devices[ddex.y].name);
	
	createContext(platforms, ddex);
	
	//createProgram(platforms, ddex, "mem_rw_test.cl");
		
	createCommandQueue(platforms, ddex, CL_QUEUE_PROFILING_ENABLE);

	initMemoryObjects(platforms, ddex, 1);
	
	for(array_size = 64; array_size <= MAX_ARRAY_SIZE; array_size *= 64){
	
		buf = (cl_char *) malloc(sizeof(cl_char) * array_size);
		
			
		platforms[ddex.x].mem_objects[0] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_READ_WRITE, sizeof(char) * array_size, NULL, &ret_num);
		checkResult(platforms, ret_num, "clCreateBuffer(mem_objects)");		

		
		for(i = 0; i < TIMES_COPY; i++){
			ret_num = clEnqueueWriteBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], CL_TRUE, 0, sizeof(char) * array_size, buf, 0, NULL, &write_event[i]);
			checkResult(platforms, ret_num, "clEnqueueWriteBuffer(mem_objects)");
			while(clWaitForEvents(1, &write_event[i]) != CL_SUCCESS){
			
			}
		}
		
		for(i = 0; i < TIMES_COPY; i++){
			ret_num = clEnqueueReadBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], CL_TRUE, 0, sizeof(char) * array_size, buf, 0, NULL, &read_event[i]);
			checkResult(platforms, ret_num, "clEnqueueReadBuffer(mem_objects)");
			while(clWaitForEvents(1, &read_event[i]) != CL_SUCCESS){
			
			}
		
		}
		
		
		printf("================================\n");
		for(i = 0; i < TIMES_COPY; i++){
			ret_num = clGetEventProfilingInfo(write_event[i], CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(write_event(queued))");
			ret_num = clGetEventProfilingInfo(write_event[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &sumbit, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(write_event(sumbit))");
			ret_num = clGetEventProfilingInfo(write_event[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(write_event(start))");
			ret_num = clGetEventProfilingInfo(write_event[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &quit, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(write_event(quit))");
			
			printf("WriteBuffer(%d): %d byte\n", i, array_size);
			printf("         __________s__m__u__n\n");
			printf("  queued:%20.0llu ns.\n", queued);
			printf("  sumbit:%20.0llu ns.\n", sumbit);
			printf(" que-sum:%20.0llu ns.\n", sumbit - queued);
			printf("   start:%20.0llu ns.\n", start);
			printf(" sum-sta:%20.0llu ns.\n", start - sumbit);
			printf("    quit:%20.0llu ns.\n", quit);
			printf(" sta-qui:%20.0llu ns.\n", quit - start);
			printf("   total:%20.0llu ns.\n", quit - queued);
			
		}
		
		for(i = 0; i < TIMES_COPY; i++){
			ret_num = clGetEventProfilingInfo(read_event[i], CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(read_event(queued))");
			ret_num = clGetEventProfilingInfo(read_event[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &sumbit, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(read_event(sumbit))");
			ret_num = clGetEventProfilingInfo(read_event[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(read_event(start))");
			ret_num = clGetEventProfilingInfo(read_event[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &quit, NULL);
			checkResult(platforms, ret_num, "clGetEventProfilingInfo(read_event(quit))");

			printf("ReadBuffer(%d): %d byte\n", i, array_size);
			printf("         __________s__m__u__n\n");
			printf("  queued:%20.0llu ns.\n", queued);
			printf("  sumbit:%20.0llu ns.\n", sumbit);
			printf(" que-sum:%20.0llu ns.\n", sumbit - queued);
			printf("   start:%20.0llu ns.\n", start);
			printf(" sum-sta:%20.0llu ns.\n", start - sumbit);
			printf("    quit:%20.0llu ns.\n", quit);
			printf(" sta-qui:%20.0llu ns.\n", quit - start);
			printf("   total:%20.0llu ns.\n", quit - queued);

		}
		

		free((void*)buf);
		ret_num = clReleaseMemObject(platforms[ddex.x].mem_objects[0]);
		checkResult(platforms, ret_num, "clReleaseMemObject(mem_objects)");

			
	}		
	
	
	platforms[ddex.x].num_mems = 0;
	free((void*)platforms[ddex.x].mem_objects);
	
	cleanUp(platforms);
	
	return 0;
}
