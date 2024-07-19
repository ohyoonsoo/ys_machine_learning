#! /bin/bash
# if [[ -z $ROOT ]]; then
#     echo "Please define YS_ML_ROOT"
#     exit 1
# fi

dir = $(echo "$@" | tr a-z A-Z) # makes input all uppercase
model_name_lower = ${echo "$@" | tr A-Z a-z}

mkdir ./$dir
touch ./$dir/CMakelists.txt
touch ./$dir/"$model_name_lower.hpp"
touch ./$dir/"$model_name_lower.cpp"
