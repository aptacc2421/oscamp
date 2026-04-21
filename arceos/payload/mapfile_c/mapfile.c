#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

void create_file(const char *fname)
{
    int fd;
    int ret;
    char content[] = "hello, arceos!";

    fd = creat(fname, 0600);
    if (fd < 0) {
        printf("Create file error!\n");
        exit(-1);
    }
    ret = write(fd, content, strlen(content)+1);
    if (ret < 0) {
        printf("Write file error!\n");
        exit(-1);
    }
    close(fd);
}

void verify_file(const char *fname)
{
    int fd;
    int ret;
    char *addr = NULL;

    fd = open(fname, O_RDONLY);
    if (fd < 0) {
        printf("Open file error!\n");
        exit(-1);
    }
    addr = mmap(NULL, 32, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == NULL) {
        printf("Map file error!\n");
        exit(-1);
    }
    printf("Read back content: %s\n", addr);
    close(fd);
}

int main()
{
    int fd;
    int ret;
    char fname[] = "test_file";

    printf("MapFile ...\n");

    create_file(fname);
    verify_file(fname);

    printf("MapFile ok!\n");
    return 0;
}
