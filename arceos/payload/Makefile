SUB_DIRS=origin hello_c fileops_c mapfile_c skernel skernel2

all: $(SUB_DIRS)

$(SUB_DIRS): FORCE
	make -C $@

FORCE:

.PHONY: all SUB_DIRS FORCE
