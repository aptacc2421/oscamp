#!/bin/sh

rm -f pflash.img
rm -f disk.img
make pflash_img
make disk_img

make run A=labs/lab1 2>/dev/null | tail -n -4
