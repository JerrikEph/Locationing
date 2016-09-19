#!/bin/bash
cnt=5
src=./test_case_ans/sample_case
res=./test_case/output_case
while (($cnt <=50))
do
    python locate_min.py --src-path=${src}003_input.txt --res-path=${res}003_${cnt}.txt --gpu-num=2 --num-station=$cnt
    cnt=$((cnt + 1))
done

printf '%03d' $cnt
