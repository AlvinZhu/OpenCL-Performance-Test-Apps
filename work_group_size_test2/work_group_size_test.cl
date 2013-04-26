__kernel void work_group_size_test_kernel_1(__global char *a) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	char var1 = a[y*4096 + x];
	
	a[y*4096 + x] = var1;
}

__kernel void work_group_size_test_kernel_2(__global char *a, __global char *b) {
	int x = get_global_id(1);
	int y = get_global_id(0);

	char var1 = a[y*4096 + x];
	char var2 = b[y*4096 + x];
	
	a[y*4096 + x] = var1 + var2;
}

__kernel void work_group_size_test_kernel_3(__global char *a, __global char *b, __global char *c) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	char var1 = a[y*4096 + x];
	char var2 = b[y*4096 + x];
	char var3 = c[y*4096 + x];
	
	a[y*4096 + x] = var1 + var2 + var3;
}

__kernel void work_group_size_test_kernel_4(__global char *a, __global char *b, __global char *c, __global char *d) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	char var1 = a[y*4096 + x];
	char var2 = b[y*4096 + x];
	char var3 = c[y*4096 + x];
	char var4 = c[y*4096 + x];
	
	a[y*4096 + x] = var1 + var2 + var3 + var4;
}
