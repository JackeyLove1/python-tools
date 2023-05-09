#!/bin/bash

script_name="$(basename $0)"

find . -type f ! -name "*.py" ! -name "$script_name" -exec rm -f {} \;