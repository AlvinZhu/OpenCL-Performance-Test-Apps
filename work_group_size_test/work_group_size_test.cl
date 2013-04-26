#define GLOBAL_WORK_GROUP_SIZE	65536
#define LOCAL_SEG_SIZE	256

__kernel void kernel_1(__global char *a) {
	int x = get_global_id(0);
	int i;
	char var1;

	for (i = 0; i < LOCAL_SEG_SIZE; i++) {
		var1 = a[i*GLOBAL_WORK_GROUP_SIZE + x];

		a[i*GLOBAL_WORK_GROUP_SIZE + x] = var1;
	}
}

__kernel void kernel_2(__global char *a, __global char *b) {
	int x = get_global_id(0);
	int i;
	char var1;
	char var2;

	for (i = 0; i < LOCAL_SEG_SIZE; i++) {
		var1 = a[i*GLOBAL_WORK_GROUP_SIZE + x];
		var2 = b[i*GLOBAL_WORK_GROUP_SIZE + x];

		a[i*GLOBAL_WORK_GROUP_SIZE + x] = var1 + var2;
	}
}

__kernel void kernel_3(__global char *a, __global char *b, __global char *c) {
	int x = get_global_id(0);
	int i;
	char var1;
	char var2;
	char var3;

	for (i = 0; i < LOCAL_SEG_SIZE; i++) {
		var1 = a[i*GLOBAL_WORK_GROUP_SIZE + x];
		var2 = b[i*GLOBAL_WORK_GROUP_SIZE + x];
		var3 = b[i*GLOBAL_WORK_GROUP_SIZE + x];

		a[i*GLOBAL_WORK_GROUP_SIZE + x] = var1 + var2 + var3;
	}
}

__kernel void kernel_4(__global char *a, __global char *b, __global char *c, __global char *d) {
	int x = get_global_id(0);
	int i;
	char var1;
	char var2;
	char var3;
	char var4;
	for (i = 0; i < LOCAL_SEG_SIZE; i++) {
		var1 = a[i*GLOBAL_WORK_GROUP_SIZE + x];
		var2 = b[i*GLOBAL_WORK_GROUP_SIZE + x];
		var3 = b[i*GLOBAL_WORK_GROUP_SIZE + x];
		var4 = d[i*GLOBAL_WORK_GROUP_SIZE + x];
		a[i*GLOBAL_WORK_GROUP_SIZE + x] = var1 + var2 + var3 + var4;
	}
}
