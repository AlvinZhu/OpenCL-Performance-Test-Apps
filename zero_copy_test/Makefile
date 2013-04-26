
SRCS := $(wildcard *.c)
OBJS := ${SRCS:%.c=%.o}
DEPS := ${OBJS:%.o=%.d}
#TARGETS := $(basename $(SRCS))
TARGETS := $(shell basename `pwd`)


WARN_FLAGS := -Wall
#OPT_FLAGS := -g 
#INCLUDES := -I/usr/local/cuda/include
#LIBRARY_PATH := -L/usr/local/cuda-5.0/lib64

INCLUDES := -I/opt/AMDAPP/include
LIBRARY_PATH := -L/opt/AMDAPP/lib/x86_64
LIB := -lOpenCL -lm

CFLAGS := $(WARN_FLAGS) $(OPT_FLAGS) $(INCLUDES)
LDFLAGS := $(LIBRARY_PATH) $(LIB)

CC 			:= gcc
LD			:= gcc
RM			:= rm -rf


all:$(TARGETS)

$(TARGETS):$(OBJS) alvincl.o
	$(LD) $(CFLAGS) -o $@ $^ $(LDFLAGS)

#%.o:%.c
#	$(CC) $(CFLAGS) -c -o $@ $<

%.d: %.c
	$(CC) -MM $(CFLAGS) $< > $@.$$$$; \
	sed 's,/($*/)/.o[ :]*,/1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

sinclude $(DEPS)

alvincl.o:alvincl/alvincl.c
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY:	clean
clean:
	rm -rf $(TARGETS) $(OBJS) $(DEPS) alvincl.o

