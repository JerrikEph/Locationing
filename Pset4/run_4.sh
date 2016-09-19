#!/bin/bash
cnt=26
src=../compete_data/case
res=./compete_case/output_case
while (($cnt <=30))
do
    python locate.py --src-path=${src}$(printf '%03d' $cnt)_input.txt --res-path=${res}$(printf '%03d' $cnt).txt --gpu-num=0
    cnt=$((cnt + 1))
done

printf '%03d' $cnt
