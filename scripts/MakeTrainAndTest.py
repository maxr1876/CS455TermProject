import sys
from os.path import basename
import os

if len(sys.argv) < 2:
	print("Please provide name of root directory of files to be split")
rootdir = sys.argv[1]
# f = open(sys.argv[1], "r")
# now you can call it directly with basename
for subdir, dirs, files in os.walk(rootdir):
    # test = open(subdir.split(" ")[1]+"_test", "w+")
    for file in files:
    	test = open(rootdir+"/"+file.split("_")[0]+"_test", "w+")
    	train = open(rootdir+"/"+file.split("_")[0]+"_train", "w+")
    	lines = open(rootdir+"/"+file, "r").readlines()
    	i = 0
    	for line in lines:
    		i = i+1
    		if i > .75 * len(lines):
    			test.write(line)
    		else:
    			train.write(line)
