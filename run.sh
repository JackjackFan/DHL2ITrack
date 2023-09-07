#!/bin/bash
export PYTHONPATH=/home/cv/data/multi_mod/RT-MDNet/re-write-RT-MDNet/RT-MDNet
python tracker.py &
sleep 2
python tracker1.py&
sleep 2
python tracker2.py&
sleep 2
