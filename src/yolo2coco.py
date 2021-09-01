#!/usr/bin/env python

import os
import sys
from pathlib import Path
import ijson
import imagesize
import csv

bdd100k_out = sys.argv[1]
#out = sys.argv[2]
#filter = sys.argv[3]

image_id = 0
annotation_id = 0

for m in ["trains", "valids"]:
    file_names = Path(bdd100k_out + "/images/" + m).glob('*.jpg')
    
    d={}
    d["images"]=[]
    d["annotations"]=[]
    
    for file_name in file_names:
        width, height = imagesize.get(file_name)
        d["images"].append({
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": image_id
        })
        with open(str(file_name).replace('images','labels').replace('jpg','txt'), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            l = [row for row in reader]
            for i in l:
                category_id=i[0]
                d["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox":[
                    ],
                    
                    
                })
                annotation_id = annotation_id + 1
#            f.read().split()
        
        image_id = image_id + 1
#    with open(m + '.json', 'w') as f:
#        json.dump(d, f, indent=4)
