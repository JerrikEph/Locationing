#!/bin/bash
cnt=5
src=./test_case_ans/sample_case
res=./test_case/output_case
while (($cnt <=60))
do
    python locate_min.py --src-path=${src}004_input.txt --res-path=${res}004_${cnt}.txt --gpu-num=3 --num-station=$cnt
    cnt=$((cnt + 1))
done

printf '%03d' $cnt
