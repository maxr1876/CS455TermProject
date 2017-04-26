import sys
totLines = 0.0
numCorrect = 0.0
for line in open(sys.argv[1], "r"):
	line = line.replace("(", "")
	line = line.replace(")", "")
	line = line.replace(" ", "")
	totLines = totLines + 1
	if float(line.split(",")[0]) == float(line.split(",")[1]):
		numCorrect = numCorrect + 1

percent = numCorrect/totLines
print "Accuracy of predictions: " + str(numCorrect/totLines)
