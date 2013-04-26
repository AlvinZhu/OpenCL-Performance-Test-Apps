#include <stdio.h>
#include <stdlib.h>

#include "alvincl/alvincl.h"

#include <CL/cl_ext.h>

#define DEVICE_TYPE				CL_DEVICE_TYPE_GPU
#define MAX_ARRAY_SIZE			16777216	// GLOBAL_WORK_GROUP_SIZE*LOCAL_SEG_SIZE	65536*256
#define GLOBAL_WORK_GROUP_SIZE	65536
#define LOCAL_WORK_GROUP_SIZE	64
#define NUM_ARRAY				2	// 1, 2, 3 or 4

#define MEM_ALIGNMENT			256

void profiling(platform_struct *platforms, cl_event* map_event, cl_event* unmap_event, cl_event* kernel_event, cl_event* rst_map_event, cl_event* rst_unmap_event) {
	cl_ulong queued, sumbit, start, quit;
	
	cl_int ret_num;
	
	cl_uint i;
	
	for(i = 0; i < NUM_ARRAY; i++){	
		ret_num = clGetEventProfilingInfo(map_event[i], CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
		checkResult(platforms, ret_num, "clGetEventProfilingInfo(map_event(queued))");
		ret_num = clGetEventProfilingInfo(map_event[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &sumbit, NULL);
		checkResult(platforms, ret_num, "clGetEventProfilingInfo(map_event(sumbit))");
		ret_num = clGetEventProfilingInfo(map_event[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		checkResult(platforms, ret_num, "clGetEventProfilingInfo(map_event(start))");
		ret_num = clGetEventProfilingInfo(map_event[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &quit, NULL);
		checkResult(platforms, ret_num, "clGetEventProfilingInfo(map_event(quit))");	
		printf("  map%d:%10.0llu ns.\t", i, quit - queued);

		ret_num = clGetEventProfilingInfo(unmap_event[i], CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
		checkResult(platforms, ret_num, "clGetEventProfilingInfo(unmap_event(queued))");
		ret_num = clGetEventProfilingInfo(unmap_event[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &sumbit, NULL);
		checkResult(platforms, ret_num, "clGetEventProfilingInfo(unmap_event(sumbit))");
		ret_num = clGetEventProfilingInfo(unmap_event[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		checkResult(platforms, ret_num, "clGetEventProfilingInfo(unmap_event(start))");
		ret_num = clGetEventProfilingInfo(unmap_event[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &quit, NULL);
		checkResult(platforms, ret_num, "clGetEventProfilingInfo(unmap_event(quit))");	
		printf("unmap%d:%10.0llu ns.\t", i, quit - queued);
	}
	printf("\n");
	
	ret_num = clGetEventProfilingInfo(*kernel_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(queued))");
	ret_num = clGetEventProfilingInfo(*kernel_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &sumbit, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(sumbit))");
	ret_num = clGetEventProfilingInfo(*kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(start))");
	ret_num = clGetEventProfilingInfo(*kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &quit, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(kernel_event(quit))");
	printf("kernel:%10.0llu ns.\t", quit - queued);
	
	ret_num = clGetEventProfilingInfo(*rst_map_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(rst_map_event(queued))");
	ret_num = clGetEventProfilingInfo(*rst_map_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &sumbit, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(rst_map_event(sumbit))");
	ret_num = clGetEventProfilingInfo(*rst_map_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(rst_map_event(start))");
	ret_num = clGetEventProfilingInfo(*rst_map_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &quit, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(rst_map_event(quit))");	
	printf("  rmap:%10.0llu ns.\t", quit - queued);
	
	ret_num = clGetEventProfilingInfo(*rst_unmap_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(rst_unmap_event(queued))");
	ret_num = clGetEventProfilingInfo(*rst_unmap_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &sumbit, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(rst_unmap_event(sumbit))");
	ret_num = clGetEventProfilingInfo(*rst_unmap_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(rst_unmap_event(start))");
	ret_num = clGetEventProfilingInfo(*rst_unmap_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &quit, NULL);
	checkResult(platforms, ret_num, "clGetEventProfilingInfo(rst_unmap_event(quit))");	
	printf("runmap:%10.0llu ns.\n", quit - queued);
}


void defaultTest(platform_struct *platforms, cl_uint2 ddex) {
	cl_char *buf[NUM_ARRAY];
	
	cl_char *ptr1[NUM_ARRAY];
	cl_char *ptr2;
	
	cl_event map_event[NUM_ARRAY];
	cl_event unmap_event[NUM_ARRAY];
	cl_event kernel_event;
	cl_event rst_map_event;
	cl_event rst_unmap_event;
	
	size_t global_work_group_size[1] = { GLOBAL_WORK_GROUP_SIZE };
	size_t local_work_group_size[1] = { LOCAL_WORK_GROUP_SIZE };
	
	cl_int ret_num;
	
	cl_uint i, j;

	for(i = 0; i < NUM_ARRAY; i++){	
		buf[i] = NULL;
		
		platforms[ddex.x].mem_objects[i] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_READ_WRITE, sizeof(char) * MAX_ARRAY_SIZE, NULL, &ret_num);
		checkResult(platforms, ret_num, "clCreateBuffer(mem_objects)");		
		
		buf[i]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &map_event[i], &ret_num);
		checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
		while(clWaitForEvents(1, &map_event[i]) != CL_SUCCESS){
		
		}
		ptr1[i] = buf[i];
		for(j = 0; j < MAX_ARRAY_SIZE; j++) {
			buf[i][j] = 1;
		}
		
		ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], buf[i], 0, NULL, &unmap_event[i]);
		checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
		while(clWaitForEvents(1, &unmap_event[i]) != CL_SUCCESS){
		
		}
		
		ret_num = clSetKernelArg(platforms[ddex.x].kernel, i, sizeof(cl_mem), &platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clEnqueueWriteBuffer(mem_objects)");
	}

	ret_num = clEnqueueNDRangeKernel(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].kernel, 1, NULL, global_work_group_size, local_work_group_size, 0, NULL, &kernel_event);
	checkResult(platforms, ret_num, "clEnqueueNDRangeKernel");	
	while(clWaitForEvents(1, &kernel_event) != CL_SUCCESS){
		
	}
	
	buf[0]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], CL_FALSE, CL_MAP_READ, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &rst_map_event, &ret_num);
	checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
	while(clWaitForEvents(1, &rst_map_event) != CL_SUCCESS){
	
	}	
	ptr2 = buf[0];
	for(j = 0; j < MAX_ARRAY_SIZE; j++) {
		if(buf[0][j] != NUM_ARRAY){
			fprintf(stderr, "ERROR: compute result error! %d %d\n", j, buf[0][j]);
			cleanUp(platforms);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}	
	}
	
	ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], buf[0], 0, NULL, &rst_unmap_event);
	checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
	while(clWaitForEvents(1, &rst_unmap_event) != CL_SUCCESS){
	
	}	
	
	printf("DEFAULT\n");
	if(ptr2 == ptr1[0]){
		printf("PTR1 = PTR2\n");
	}
	profiling(platforms, map_event, unmap_event, &kernel_event, &rst_map_event, &rst_unmap_event);
		
		
		
	for(i = 0; i < NUM_ARRAY; i++){
		ret_num = clReleaseMemObject(platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clReleaseMemObject(mem_objects)");
	}
}

void allocTest(platform_struct *platforms, cl_uint2 ddex) {
	cl_char *buf[NUM_ARRAY];

	cl_char *ptr1[NUM_ARRAY];
	cl_char *ptr2;
	
	cl_event map_event[NUM_ARRAY];
	cl_event unmap_event[NUM_ARRAY];
	cl_event kernel_event;
	cl_event rst_map_event;
	cl_event rst_unmap_event;
	
	size_t global_work_group_size[1] = { GLOBAL_WORK_GROUP_SIZE };
	size_t local_work_group_size[1] = { LOCAL_WORK_GROUP_SIZE };
	
	cl_int ret_num;
	
	cl_uint i, j;

	for(i = 0; i < NUM_ARRAY; i++){	
		buf[i] = NULL;
		
		platforms[ddex.x].mem_objects[i] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(char) * MAX_ARRAY_SIZE, NULL, &ret_num);
		checkResult(platforms, ret_num, "clCreateBuffer(mem_objects)");		
		
		buf[i]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &map_event[i], &ret_num);
		checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
		while(clWaitForEvents(1, &map_event[i]) != CL_SUCCESS){
		
		}
		ptr1[i] = buf[i];
		for(j = 0; j < MAX_ARRAY_SIZE; j++) {
			buf[i][j] = 1;
		}
		
		ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], buf[i], 0, NULL, &unmap_event[i]);
		checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
		while(clWaitForEvents(1, &unmap_event[i]) != CL_SUCCESS){
		
		}
		
		ret_num = clSetKernelArg(platforms[ddex.x].kernel, i, sizeof(cl_mem), &platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clEnqueueWriteBuffer(mem_objects)");
	}

	ret_num = clEnqueueNDRangeKernel(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].kernel, 1, NULL, global_work_group_size, local_work_group_size, 0, NULL, &kernel_event);
	checkResult(platforms, ret_num, "clEnqueueNDRangeKernel");	
	while(clWaitForEvents(1, &kernel_event) != CL_SUCCESS){
		
	}
	
	buf[0]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], CL_FALSE, CL_MAP_READ, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &rst_map_event, &ret_num);
	checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
	while(clWaitForEvents(1, &rst_map_event) != CL_SUCCESS){
	
	}	
	ptr2 = buf[0];
	for(j = 0; j < MAX_ARRAY_SIZE; j++) {
		if(buf[0][j] != NUM_ARRAY){
			fprintf(stderr, "ERROR: compute result error! %d %d\n", j, buf[0][j]);
			cleanUp(platforms);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}	
	}
	
	ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], buf[0], 0, NULL, &rst_unmap_event);
	checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
	while(clWaitForEvents(1, &rst_unmap_event) != CL_SUCCESS){
	
	}				
	
	printf("ALLOC\n");
	if(ptr2 == ptr1[0]){
		printf("PTR1 = PTR2\n");
	}
	profiling(platforms, map_event, unmap_event, &kernel_event, &rst_map_event, &rst_unmap_event);
	
	for(i = 0; i < NUM_ARRAY; i++){
		ret_num = clReleaseMemObject(platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clReleaseMemObject(mem_objects)");
	}
}

void useTest(platform_struct *platforms, cl_uint2 ddex) {
	cl_char *buf[NUM_ARRAY];
	
	cl_char *ptr1[NUM_ARRAY];
	cl_char *ptr2;
	
	cl_event map_event[NUM_ARRAY];
	cl_event unmap_event[NUM_ARRAY];
	cl_event kernel_event;
	cl_event rst_map_event;
	cl_event rst_unmap_event;
	
	size_t global_work_group_size[1] = { GLOBAL_WORK_GROUP_SIZE };
	size_t local_work_group_size[1] = { LOCAL_WORK_GROUP_SIZE };
	
	cl_int ret_num;
	
	cl_uint i, j;

	for(i = 0; i < NUM_ARRAY; i++){	
		//buf[i] = (cl_char *) calloc(sizeof(cl_char), MAX_ARRAY_SIZE);
		posix_memalign((void**)&buf[i], MEM_ALIGNMENT, MAX_ARRAY_SIZE);
		checkPointer(platforms, buf[i], "buf[i]");
		
		platforms[ddex.x].mem_objects[i] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(char) * MAX_ARRAY_SIZE, buf[i], &ret_num);
		checkResult(platforms, ret_num, "clCreateBuffer(mem_objects)");		
		
		buf[i]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &map_event[i], &ret_num);
		checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
		while(clWaitForEvents(1, &map_event[i]) != CL_SUCCESS){
		
		}
		ptr1[i] = buf[i];
		for(j = 0; j < MAX_ARRAY_SIZE; j++) {
			buf[i][j] = 1;
		}
		
		ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], buf[i], 0, NULL, &unmap_event[i]);
		checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
		while(clWaitForEvents(1, &unmap_event[i]) != CL_SUCCESS){
		
		}
		
		ret_num = clSetKernelArg(platforms[ddex.x].kernel, i, sizeof(cl_mem), &platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clEnqueueWriteBuffer(mem_objects)");
	}

	ret_num = clEnqueueNDRangeKernel(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].kernel, 1, NULL, global_work_group_size, local_work_group_size, 0, NULL, &kernel_event);
	checkResult(platforms, ret_num, "clEnqueueNDRangeKernel");	
	while(clWaitForEvents(1, &kernel_event) != CL_SUCCESS){
		
	}
	
	buf[0]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], CL_FALSE, CL_MAP_READ, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &rst_map_event, &ret_num);
	checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
	while(clWaitForEvents(1, &rst_map_event) != CL_SUCCESS){
	
	}	
	ptr2 = buf[0];
	for(j = 0; j < MAX_ARRAY_SIZE; j++) {
		if(buf[0][j] != NUM_ARRAY){
			fprintf(stderr, "ERROR: compute result error! %d %d\n", j, buf[0][j]);
			cleanUp(platforms);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}	
	}
	
	ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], buf[0], 0, NULL, &rst_unmap_event);
	checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
	while(clWaitForEvents(1, &rst_unmap_event) != CL_SUCCESS){
	
	}	
			
	
	printf("USE\n");
	if(ptr2 == ptr1[0]){
		printf("PTR1 = PTR2\n");
	}
	profiling(platforms, map_event, unmap_event, &kernel_event, &rst_map_event, &rst_unmap_event);
	
		
	for(i = 0; i < NUM_ARRAY; i++){
		ret_num = clReleaseMemObject(platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clReleaseMemObject(mem_objects)");
	}
}

void copyTest(platform_struct *platforms, cl_uint2 ddex) {
	cl_char *buf[NUM_ARRAY];
	
	cl_char *ptr1[NUM_ARRAY];
	cl_char *ptr2;
	
	cl_event map_event[NUM_ARRAY];
	cl_event unmap_event[NUM_ARRAY];
	cl_event kernel_event;
	cl_event rst_map_event;
	cl_event rst_unmap_event;
	
	size_t global_work_group_size[1] = { GLOBAL_WORK_GROUP_SIZE };
	size_t local_work_group_size[1] = { LOCAL_WORK_GROUP_SIZE };
	
	cl_int ret_num;
	
	cl_uint i, j;

	for(i = 0; i < NUM_ARRAY; i++){	
		//buf[i] = (cl_char *) calloc(sizeof(cl_char), MAX_ARRAY_SIZE);
		posix_memalign((void**)&buf[i], MEM_ALIGNMENT, MAX_ARRAY_SIZE);
		checkPointer(platforms, buf[i], "buf[i]");
		
		platforms[ddex.x].mem_objects[i] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(char) * MAX_ARRAY_SIZE, buf[i], &ret_num);
		checkResult(platforms, ret_num, "clCreateBuffer(mem_objects)");		
		
		buf[i]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &map_event[i], &ret_num);
		checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
		while(clWaitForEvents(1, &map_event[i]) != CL_SUCCESS){
		
		}
		ptr1[i] = buf[i];
		for(j = 0; j < MAX_ARRAY_SIZE; j++) {
			buf[i][j] = 1;
		}
		
		ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], buf[i], 0, NULL, &unmap_event[i]);
		checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
		while(clWaitForEvents(1, &unmap_event[i]) != CL_SUCCESS){
		
		}
		
		ret_num = clSetKernelArg(platforms[ddex.x].kernel, i, sizeof(cl_mem), &platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clEnqueueWriteBuffer(mem_objects)");
	}

	ret_num = clEnqueueNDRangeKernel(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].kernel, 1, NULL, global_work_group_size, local_work_group_size, 0, NULL, &kernel_event);
	checkResult(platforms, ret_num, "clEnqueueNDRangeKernel");	
	while(clWaitForEvents(1, &kernel_event) != CL_SUCCESS){
		
	}
	
	buf[0]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], CL_FALSE, CL_MAP_READ, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &rst_map_event, &ret_num);
	checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
	while(clWaitForEvents(1, &rst_map_event) != CL_SUCCESS){
	
	}	
	ptr2 = buf[0];
	for(j = 0; j < MAX_ARRAY_SIZE; j++) {
		if(buf[0][j] != NUM_ARRAY){
			fprintf(stderr, "ERROR: compute result error! %d %d\n", j, buf[0][j]);
			cleanUp(platforms);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}	
	}
	
	ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], buf[0], 0, NULL, &rst_unmap_event);
	checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
	while(clWaitForEvents(1, &rst_unmap_event) != CL_SUCCESS){
	
	}	
			
	
	printf("COPY\n");
	if(ptr2 == ptr1[0]){
		printf("PTR1 = PTR2\n");
	}
	profiling(platforms, map_event, unmap_event, &kernel_event, &rst_map_event, &rst_unmap_event);

	for(i = 0; i < NUM_ARRAY; i++){
		ret_num = clReleaseMemObject(platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clReleaseMemObject(mem_objects)");
	}
}

void amdTest(platform_struct *platforms, cl_uint2 ddex) {
	cl_char *buf[NUM_ARRAY];
	
	cl_char *ptr1[NUM_ARRAY];
	cl_char *ptr2;
	
	cl_event map_event[NUM_ARRAY];
	cl_event unmap_event[NUM_ARRAY];
	cl_event kernel_event;
	cl_event rst_map_event;
	cl_event rst_unmap_event;

	size_t global_work_group_size[1] = { GLOBAL_WORK_GROUP_SIZE };
	size_t local_work_group_size[1] = { LOCAL_WORK_GROUP_SIZE };
	
	cl_int ret_num;
	
	cl_uint i, j;

	for(i = 0; i < NUM_ARRAY; i++){	
		buf[i] = NULL;
		
		platforms[ddex.x].mem_objects[i] = clCreateBuffer(platforms[ddex.x].context, CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD, sizeof(char) * MAX_ARRAY_SIZE, NULL, &ret_num);
		checkResult(platforms, ret_num, "clCreateBuffer(mem_objects)");		
		
		buf[i]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &map_event[i], &ret_num);
		checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
		while(clWaitForEvents(1, &map_event[i]) != CL_SUCCESS){
		
		}
		ptr1[i] = buf[i];
		for(j = 0; j < MAX_ARRAY_SIZE; j++) {
			buf[i][j] = 1;
		}
		
		ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[i], buf[i], 0, NULL, &unmap_event[i]);
		checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
		while(clWaitForEvents(1, &unmap_event[i]) != CL_SUCCESS){
		
		}
		
		ret_num = clSetKernelArg(platforms[ddex.x].kernel, i, sizeof(cl_mem), &platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clEnqueueWriteBuffer(mem_objects)");
	}

	ret_num = clEnqueueNDRangeKernel(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].kernel, 1, NULL, global_work_group_size, local_work_group_size, 0, NULL, &kernel_event);
	checkResult(platforms, ret_num, "clEnqueueNDRangeKernel");	
	while(clWaitForEvents(1, &kernel_event) != CL_SUCCESS){
		
	}
	
	buf[0]= clEnqueueMapBuffer(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], CL_FALSE, CL_MAP_READ, 0, sizeof(char) * MAX_ARRAY_SIZE, 0, NULL, &rst_map_event, &ret_num);
	checkResult(platforms, ret_num, "clEnqueueMapBuffer(mem_objects)");
	while(clWaitForEvents(1, &rst_map_event) != CL_SUCCESS){
	
	}	
	ptr2 = buf[0];
	for(j = 0; j < MAX_ARRAY_SIZE; j++) {
		if(buf[0][j] != NUM_ARRAY){
			fprintf(stderr, "ERROR: compute result error! %d %d\n", j, buf[0][j]);
			cleanUp(platforms);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}	
	}
	
	ret_num = clEnqueueUnmapMemObject(platforms[ddex.x].devices[ddex.y].command_queue, platforms[ddex.x].mem_objects[0], buf[0], 0, NULL, &rst_unmap_event);
	checkResult(platforms, ret_num, "clEnqueueUnmapMemObject(mem_objects)");
	while(clWaitForEvents(1, &rst_unmap_event) != CL_SUCCESS){
	
	}	
			
	
	printf("AMD\n");
	if(ptr2 == ptr1[0]){
		printf("PTR1 = PTR2\n");
	}
	profiling(platforms, map_event, unmap_event, &kernel_event, &rst_map_event, &rst_unmap_event);

	for(i = 0; i < NUM_ARRAY; i++){
		ret_num = clReleaseMemObject(platforms[ddex.x].mem_objects[i]);
		checkResult(platforms, ret_num, "clReleaseMemObject(mem_objects)");
	}
}


int main() {
	cl_uint num_platforms;
	platform_struct *platforms = NULL;
	cl_uint2 ddex;
	
	cl_int ret_num;	
	
	
	platforms = getPlatforms(&num_platforms);
	getDevices(platforms);
	
	ddex = setDevice(platforms, DEVICE_TYPE);
	
	printf("select device %s \n", platforms[ddex.x].devices[ddex.y].name);
	
	createContext(platforms, ddex);
	
	createProgram(platforms, ddex, "zero_copy_test.cl");
	
	char kernelname[1024];
	sprintf(kernelname, "kernel_%d", NUM_ARRAY);
	platforms[ddex.x].kernel = clCreateKernel(platforms[ddex.x].program, kernelname, &ret_num);
	checkResult(platforms, ret_num, "clCreateKernel");
		
	createCommandQueue(platforms, ddex, CL_QUEUE_PROFILING_ENABLE);

	initMemoryObjects(platforms, ddex, NUM_ARRAY);
	
	defaultTest(platforms, ddex);
	copyTest(platforms, ddex);
	allocTest(platforms, ddex);
	useTest(platforms, ddex);
	amdTest(platforms, ddex);
	
	platforms[ddex.x].num_mems = 0;
	free((void*)platforms[ddex.x].mem_objects);
	
	cleanUp(platforms);
	
	return 0;
}
