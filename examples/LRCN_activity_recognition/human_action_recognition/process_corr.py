import os
import sys
import numpy as np
import glob
from scipy.signal import argrelextrema

corr_file = '/media/anhnguyen/Data/lisa-caffe-public/examples/LRCN_activity_recognition/video_corr.txt'
frame_folder = '/media/anhnguyen/4C35-FA26/UCF101-split1/'

# Read corr file
f = open(corr_file, 'r')
f_lines = f.readlines()
f.close()

f_out = open('ucf101_split1_testVideos_minima_maxima.txt', 'w')

# Extract video, label and corr array
video_dict = {}

for line in f_lines:
	line = line.split(' ')
	video = line[0]
	label = line[1]
	corr = []
	for item in line[2:len(line)-1]:
		corr.append(float(item))

	print video + ' ' + str(label) + ' ' + str(len(corr))

	frames = glob.glob('%s%s/*.jpg' %(frame_folder, video))
	frames.sort()
	total_frames = len(frames)

	# Find global minimum
	corr = np.array(corr)
	negative_peak = np.argmin(np.array(corr))

	# Find first significant maximum
	maxm = argrelextrema(np.array(corr), np.greater)
	global_min = negative_peak
	selected_max = -1
	for local_max in maxm[0]:
		if local_max-global_min >= 7:
			selected_max = local_max
			break
	if selected_max == -1:
		selected_max = total_frames-2

	f_out.writelines("%s " %video)
	f_out.writelines("%s " %label)
	f_out.writelines("%d " %negative_peak)
	f_out.writelines("%d\n" %selected_max)

f_out.close()