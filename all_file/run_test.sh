#!/bin/bash
cnt=5
src=./test_case_ans/sample_case
res=./test_case/output_case
while (($cnt <=5))
do
    python locate_test.py --src-path=${src}$(printf '%03d' $cnt)_input.txt --res-path=${res}$(printf '%03d' $cnt).txt --gpu-num=0
    cnt=$((cnt + 1))
done

printf '%03d' $cnt
