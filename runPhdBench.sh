#!/usr/bin/env bash
NOW=$(date +"%Y-%m-%d-%H-%M")
HOSTNAME=$(hostname)

lineSteps=( 10 50 100 200 500 750 1000)

angleSteps=( 20 10 5 2 1 0.5 0.2 0.1 0.05 )

firstRun=true

for i in "${lineSteps[@]}"; do
	export LH_READ_N_LINES=$i
	for k in "${angleSteps[@]}"; do
		export LH_ANGLE_STEPSIZE=$k
		if [[ $firstRun==true ]]; then
			./phdBench >> /dev/null
			firstRun=false
		fi
		for l in {1..5}; do
			./phdBench >> $HOSTNAME"-"$NOW".csv"
		done
	done
done
