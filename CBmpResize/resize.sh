#!/bin/bash
inputPath=$1
outputPath=$2
type=$3

files=$(ls $inputPath)
for fileName in $files
do
    if [ "${fileName##*.}"x = "bmp"x ];then
        time ./CBmpResize "$inputPath/$fileName" "$outputPath/$fileName" $type
    fi
done
