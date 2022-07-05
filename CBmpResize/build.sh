#!/bin/sh
A="$1"
cmake -H"./" -B"./.build" -DCMAKE_BUILD_TYPE=MinSizeRel -G "Ninja"
if [[ "$A" == "update" ]];
then
    cp compile_commands.json ../
    cp CBmpResize ../
    exit 0
else
    cd .build
    ninja all
    cp compile_commands.json ../
    cp CBmpResize ../
fi
