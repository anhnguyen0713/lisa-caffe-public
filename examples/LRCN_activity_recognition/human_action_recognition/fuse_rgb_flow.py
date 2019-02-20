import sys
import os
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

#Parse program arguments
parser = argparse.ArgumentParser(description='Parameters for recognition system')
parser.add_argument('-rgb', metavar='RGB_FILE', type=str)
parser.add_argument('-flow', metavar='FLOW_FILE', type=str)
parser.add_argument('-w', metavar='WEIGHT_RGB', type=float)

args = parser.parse_args()

if not args.rgb:
	args.test = 'sample_rgb.txt'
if not args.flow:
	args.o = 'sample_flow.txt'

RGB_file = args.rgb
flow_file = args.flow
w_rgb = args.w
w_flow = 1-w_rgb

num_classes = 101

f_rgb = open(RGB_file, 'r')
f_rgb_lines = f_rgb.readlines()
f_rgb.close()

f_flow = open(flow_file, 'r')
f_flow_lines = f_flow.readlines()
f_flow.close()

num_videos = len(f_rgb_lines)

print('INFO -------------------------------------------------')
print('RGB file: ' + RGB_file)
print('Flow file: ' + flow_file)
print('RGB weight: ' + str(w_rgb))
print('Flow weight: ' + str(w_flow))
print('Num videos: ' + str(num_videos))
print('------------------------------------------------------')

rgb_probs = []
flow_probs = []
true_labels = []
predicted_labels = []

for i in range(0, num_videos):
	line_rgb = f_rgb_lines[i]
	line_rgb = line_rgb.split()
	video = line_rgb[0]
	if i<10:
		print str(i) + ' ' + video
	label = int(line_rgb[1])
	rgb_prob = np.array(line_rgb[2:num_classes+2]).astype(np.float32)

	line_flow = f_flow_lines[i]
	line_flow = line_flow.split()
	video = line_flow[0]
	label = int(line_flow[1])
	flow_prob = np.array(line_flow[2:num_classes+2]).astype(np.float32)

	true_labels.append(label)
	
	rgb_prob = np.multiply(rgb_prob, w_rgb)

	flow_prob = np.multiply(flow_prob, w_flow)

	fused_prob = np.add(rgb_prob, flow_prob)
	predicted_label, = np.where(fused_prob == np.max(fused_prob))
	predicted_labels.append(predicted_label[0])

print true_labels[0:10]
print predicted_labels[0:10]
print 'Accuracy: ' + str(accuracy_score(true_labels, predicted_labels)*100) + '%'
