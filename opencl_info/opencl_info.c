#include <stdio.h>
#include <stdlib.h>

#include "alvincl/alvincl.h"

cl_char* GetPlatformInfo(platform_struct *platform, cl_platform_info param_name) {
	size_t param_value_size;	
	cl_char* param_value;
	
	cl_int ret_num;
	
	ret_num = clGetPlatformInfo(platform->id, param_name, (size_t) 0, NULL, (size_t *) &param_value_size);
	checkResult(platform, ret_num, "clGetPlatformInfo(param_value_size)");
		
	param_value = (cl_char *) malloc(param_value_size+1);
	checkPointer(platform, param_value, "platform.name");
	
	ret_num = clGetPlatformInfo(platform->id, param_name, param_value_size, param_value, (size_t *) NULL);
	checkResult(platform, ret_num, "clGetPlatformInfo(param_value)");
	
	param_value[param_value_size] = '\0';
	
	return param_value;
}

cl_char* getDeviceInfo(platform_struct *platform, cl_device_id id , cl_device_info param_name) {
	size_t param_value_size;	
	cl_char* param_value;
	
	cl_int ret_num;
	
	ret_num = clGetDeviceInfo(id, param_name, (size_t) 0, NULL, (size_t *) &param_value_size);
	checkResult(platform, ret_num, "clGetDeviceInfo(param_value_size)");
	
	param_value = (cl_char *) malloc(param_value_size + 1);
	checkPointer(platform, param_value, "param_value");
	
	ret_num = clGetDeviceInfo(id, param_name, param_value_size, param_value, (size_t *) NULL);
	checkResult(platform, ret_num, "clGetDeviceInfo(param_value)");
	
	param_value[param_value_size] = '\0';	 
	
	return param_value;
}

int main(int argc, char *argv[]) {
	cl_uint num_platforms;
	platform_struct *platforms = NULL;
	
	cl_int ret_num;
	
	cl_uint i, j, k;
	
	platforms = getPlatforms(&num_platforms);
	getDevices(platforms);
	
	printf("Number of platforms: %d\n", num_platforms);
	for(i = 0; i < num_platforms; i++) {
		printf("-Platforms %d:\n", i);

		printf("  CL_PLATFORM_NAME      : %s\n", platforms[i].name);
		printf("  CL_PLATFORM_VENDOR    : %s\n", GetPlatformInfo(&platforms[i], CL_PLATFORM_VENDOR));		
		printf("  CL_PLATFORM_VERSION   : %s\n", GetPlatformInfo(&platforms[i], CL_PLATFORM_VERSION));
		printf("  CL_PLATFORM_PROFILE   : %s\n", GetPlatformInfo(&platforms[i], CL_PLATFORM_PROFILE));
		printf("  CL_PLATFORM_EXTENSIONS: %s\n", GetPlatformInfo(&platforms[i], CL_PLATFORM_EXTENSIONS));	
		
		printf("  Number of devices     : %d\n", platforms[i].num_devices);
		for(j = 0; j < platforms[i].num_devices; j++) {
			printf("---Devices %d:\n", j);
			
			printf("    CL_DEVICE_NAME                         : %s\n", platforms[i].devices[j].name);

			switch(platforms[i].devices[j].type) {
				case CL_DEVICE_TYPE_DEFAULT:
					printf("    CL_DEVICE_TYPE                         : CL_DEVICE_TYPE_DEFAULT\n");
					break;
				case CL_DEVICE_TYPE_CPU:
					printf("    CL_DEVICE_TYPE                         : CL_DEVICE_TYPE_CPU\n");
					break;
				case CL_DEVICE_TYPE_GPU:
					printf("    CL_DEVICE_TYPE                         : CL_DEVICE_TYPE_GPU\n");
					break;
				case CL_DEVICE_TYPE_ACCELERATOR:
					printf("    CL_DEVICE_TYPE                         : CL_DEVICE_TYPE_ACCELERATOR\n");
					break;
				//case CL_DEVICE_TYPE_CUSTOM:
				//	printf("    CL_DEVICE_TYPE                         : CL_DEVICE_TYPE_CUSTOM\n");
				//	break;
				default:
					printf("    CL_DEVICE_TYPE                         : CL_DEVICE_TYPE_ALL\n");				
			}

			cl_uint vendor_id;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_VENDOR_ID)");
			printf("    CL_DEVICE_VENDOR_ID                    : %d\n", vendor_id);
			
			cl_uint max_compute_units;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");
			printf("    CL_DEVICE_MAX_COMPUTE_UNITS            : %d\n", max_compute_units);
		
			cl_uint max_work_item_dimensions;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dimensions, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)");
			printf("    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS     : %d\n", max_work_item_dimensions);
			
			size_t max_work_group_size;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)");
			printf("    CL_DEVICE_MAX_WORK_GROUP_SIZE          : %d\n", (cl_uint)max_work_group_size);
			
			size_t max_work_item_sizes[max_work_item_dimensions];
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, &max_work_item_sizes, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES)");
			printf("    CL_DEVICE_MAX_WORK_ITEM_SIZES          : (%d", (cl_uint)max_work_item_sizes[0]);
			for(k = 1; k < max_work_item_dimensions; k++) {
				printf(", %d", (cl_uint)max_work_item_sizes[k]);
			}
			printf(")\n");
			
			cl_uint max_clock_frequency;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_clock_frequency, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY)");
			printf("    CL_DEVICE_MAX_CLOCK_FREQUENCY          : %d MHz\n", max_clock_frequency);

			cl_uint address_bits;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &address_bits, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_ADDRESS_BITS)");
			printf("    CL_DEVICE_ADDRESS_BITS                 : %d bits\n", address_bits);

			cl_ulong max_mem_alloc_size;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE)");
			printf("    CL_DEVICE_MAX_MEM_ALLOC_SIZE           : %llu bytes\n", max_mem_alloc_size);
			
			cl_uint mem_base_addr_align;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &mem_base_addr_align, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN)");
			printf("    CL_DEVICE_MEM_BASE_ADDR_ALIGN          : %d bits\n", mem_base_addr_align);

			cl_uint min_data_type_align_size;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(cl_uint), &min_data_type_align_size, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE)");
			printf("    CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE     : %d bits\n", min_data_type_align_size);

			cl_device_mem_cache_type global_mem_cache_type;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cl_device_mem_cache_type), &global_mem_cache_type, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE)");
			switch(global_mem_cache_type) {
				case CL_READ_ONLY_CACHE:
					printf("    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE        : CL_READ_ONLY_CACHE\n");
					break;
				case CL_READ_WRITE_CACHE:
					printf("    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE        : CL_READ_WRITE_CACHE\n");
					break;
				case CL_NONE:
					printf("    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE        : CL_NONE\n");				
			}

			cl_ulong global_mem_cache_size;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &global_mem_cache_size, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)");
			printf("    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE        : %llu bytes\n", global_mem_cache_size);

			cl_ulong global_mem_size;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE)");
			printf("    CL_DEVICE_GLOBAL_MEM_SIZE              : %llu bytes\n", global_mem_size);

			cl_ulong max_constant_buffer_size;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &max_constant_buffer_size, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)");
			printf("    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE     : %llu bytes\n", max_constant_buffer_size);

			cl_device_local_mem_type local_mem_type;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &local_mem_type, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_LOCAL_MEM_TYPE)");
			switch(local_mem_type) {
				case CL_LOCAL:
					printf("    CL_DEVICE_LOCAL_MEM_TYPE               : CL_LOCAL\n");
					break;
				case CL_GLOBAL:
					printf("    CL_DEVICE_LOCAL_MEM_TYPE               : CL_GLOBAL\n");
					break;
				case CL_NONE:
					printf("    CL_DEVICE_LOCAL_MEM_TYPE               : CL_NONE\n");				
			}
			
			cl_ulong local_mem_size;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_LOCAL_MEM_SIZE)");
			printf("    CL_DEVICE_LOCAL_MEM_SIZE               : %llu bytes\n", local_mem_size);			
			
			size_t profiling_timer_resolution;
			ret_num = clGetDeviceInfo(platforms[i].devices[j].id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t), &profiling_timer_resolution, (size_t *) NULL);
			checkResult(platforms, ret_num, "clGetDeviceInfo(CL_DEVICE_PROFILING_TIMER_RESOLUTION)");
			printf("    CL_DEVICE_PROFILING_TIMER_RESOLUTION   : %d\n", (cl_uint)profiling_timer_resolution);

 			//printf("    CL_DEVICE_BUILT_IN_KERNELS             : %s\n", getDeviceInfo(&platforms[i], platforms[i].devices[j].id, CL_DEVICE_BUILT_IN_KERNELS));
 			printf("    CL_DEVICE_VENDOR                       : %s\n", getDeviceInfo(&platforms[i], platforms[i].devices[j].id, CL_DEVICE_VENDOR));
 			printf("    CL_DRIVER_VERSION                      : %s\n", getDeviceInfo(&platforms[i], platforms[i].devices[j].id, CL_DRIVER_VERSION));
 			printf("    CL_DEVICE_PROFILE                      : %s\n", getDeviceInfo(&platforms[i], platforms[i].devices[j].id, CL_DEVICE_PROFILE));
 			printf("    CL_DEVICE_EXTENSIONS                   : %s\n", getDeviceInfo(&platforms[i], platforms[i].devices[j].id, CL_DEVICE_EXTENSIONS));
 			printf("    CL_DEVICE_OPENCL_C_VERSION             : %s\n", getDeviceInfo(&platforms[i], platforms[i].devices[j].id, CL_DEVICE_OPENCL_C_VERSION));

			printf("\n");
		}
		printf("\n");
	}
	
	cleanUp(platforms);
	
	return 0;
}
