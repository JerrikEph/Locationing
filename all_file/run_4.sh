#!/bin/bash
cnt=21
src=./compete_data/case
res=./compete_case/output_case
while (($cnt <=25))
do
    python locate_3.py --src-path=${src}$(printf '%03d' $cnt)_input.txt --res-path=${res}$(printf '%03d' $cnt).txt --gpu-num=3
    cnt=$((cnt + 1))
done

printf '%03d' $cnt
