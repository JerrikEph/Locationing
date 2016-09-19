#!/bin/bash
cnt=11
src=../compete_data/case
res=./compete_case/output_case
while (($cnt <=20))
do
    python locate_min.py --src-path=${src}$(printf '%03d' $cnt)_input.txt --res-path=${res}$(printf '%03d' $cnt).txt --gpu-num=2 --num-station=10
    cnt=$((cnt + 1))
done

printf '%03d' $cnt
