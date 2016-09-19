#!/bin/bash
cnt=5
src=./test_case_ans/sample_case
res=./test_case/output_case
while (($cnt <=40))
do
    python locate_min.py --src-path=${src}002_input.txt --res-path=${res}002_${cnt}.txt --gpu-num=1 --num-station=$cnt
    cnt=$((cnt + 1))
done

printf '%03d' $cnt
