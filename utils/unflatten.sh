#!/bin/bash
# replaces all "+" with "/" 

if [[ $# -eq 0 ]]; then
	echo "Must pass argument of parent directory containing files to unflatten"
	exit 1
fi

rep="/"

for f in $1/*.jpg; do
	# replace all (//) occurences of + (+) with / (\/)
	unflattened="${f//+/$rep}"
	# echo $unflattened
	mkdir -p `dirname $unflattened`
	mv $f $unflattened

done
