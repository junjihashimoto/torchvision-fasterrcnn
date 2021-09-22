#!/usr/bin/env python
import sys
import json
import re

logfile = sys.argv[1]
file1 = open(logfile, 'r')
Lines = file1.readlines()

count = 0
map50 = -1
ap50 = []
for line in Lines:
    m = re.match(r'.* category : ([0-9]+) : ([^ ]+).*',line)
    if m:
        print("Category %d %.3f" % (int(m.group(1)),float(m.group(2))))
        ap50.append(float(m.group(2)))
    m = re.match(r' Average Precision  \(AP\) @\[ IoU=0.50      \| area=   all \| maxDets=100 \] = ([^ ]*).*',line)
    if m:
        map50 = float(m.group(1))
        print("mAP %.3f" % (map50))
        

map_f = 'map_results.json'
print('mAP ... saving %s...' % map_f)
with open(map_f, 'w') as file:
    json.dump({
        'mAP@.5': map50,
        'AP@.5': ap50
    }, file)
