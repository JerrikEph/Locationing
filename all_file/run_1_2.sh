#!/bin/bash
cnt=6
src=./compete_data/case
res=./compete_case/output_case
while (($cnt <=10))
do
    python locate.py --src-path=${src}$(printf '%03d' $cnt)_input.txt --res-path=${res}$(printf '%03d' $cnt).txt --gpu-num=2
    cnt=$((cnt + 1))
done

printf '%03d' $cnt
