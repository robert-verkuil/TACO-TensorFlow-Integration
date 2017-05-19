#!/bin/bash

export LD_PRELOAD=/usr/lib/libprofiler.so
export CPUPROFILE="profiletest.data python"
$@
