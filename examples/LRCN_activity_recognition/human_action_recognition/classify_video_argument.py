from __future__ import print_function
import numpy as np
import glob
caffe_root = '../../../'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
from sklearn.metrics import accuracy_score
import time
import math
import argparse

RGB_video_path = '/media/anhnguyen/12F29C5AF29C43BF/Motion_recognition/Low_quality/UCF101-CRF-50/'
flow_video_path = '/media/anhnguyen/Data/lisa-caffe-public/UCF-101/flow_images/'


################ LIST OF PARAMETERS #############################################
# Params 		Meaning							Values				#
# -test 	file of test videos 				xxx.txt				#
# -o		name of output file					xxx.txt				#
# -c		clip length 						integer				#
# -m		type of input 						'rgb', 'flow'		#
# -l		number of overlap frames 			integer				#
# -s		starting frame of input segment		'0', 'gmin', 'lmax'	#
# -e		end frame of input segment			'0', 'gmin', 'lmax', 'vlength'	#
#################################################################################


#Parse program arguments
parser = argparse.ArgumentParser(description='Parameters for recognition system')
parser.add_argument('-test', metavar='TEST_FILE', type=str)
parser.add_argument('-o', metavar='OUT_FILE', type=str)
parser.add_argument('-c', metavar='CLIP_LENGTH', type=int)
parser.add_argument('-m', metavar='MODALITY', type=str)
parser.add_argument('-l', metavar='OVERLAP', type=int)
parser.add_argument('-s', metavar='START_FRAME', type=str)
parser.add_argument('-e', metavar='END_FRAME', type=str)

args = parser.parse_args()

if not args.test:
	args.test = 'ucf101_split1_testVideos_minima_maxima.txt'
if not args.o:
	args.o = 'sample_results.txt'
if not args.c:
	args.c = 16
if not args.m:
	args.m = 'rgb'
if not args.l:
	args.l = 0
if not args.s:
	args.s = '0'
if not args.e:
	args.e = 'lmax'

TEST_FILE = args.test
OUT_FILE = args.o
CLIP_LENGTH = args.c
MODALITY = args.m
OVERLAP = args.l
START_FRAME = args.s
END_FRAME = args.e

f_out = open('results/' + OUT_FILE, 'w')

#Initialize transformers 
def initialize_transformer(image_mean, is_flow):
	shape = (10*CLIP_LENGTH, 3, 227, 227)
	transformer = caffe.io.Transformer({'data': shape})
	channel_mean = np.zeros((3,227,227))
	for channel_index, mean_val in enumerate(image_mean):
		channel_mean[channel_index, ...] = mean_val
	transformer.set_mean('data', channel_mean)
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2, 1, 0))
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_is_flow('data', is_flow)
	return transformer

ucf_mean_RGB = np.zeros((3,1,1))
ucf_mean_flow = np.zeros((3,1,1))
ucf_mean_flow[:,:,:] = 128
ucf_mean_RGB[0,:,:] = 103.939
ucf_mean_RGB[1,:,:] = 116.779
ucf_mean_RGB[2,:,:] = 128.68

transformer_RGB = initialize_transformer(ucf_mean_RGB, False)
transformer_flow = initialize_transformer(ucf_mean_flow,True)

def all_frame_sample(CLIP_LENGTH, start_point, end_point, global_min=0):
	selected_indices = []

	selected_indices.extend(range(start_point, end_point+1))

	while(len(selected_indices)%CLIP_LENGTH != 0):
		selected_indices.append(end_point)

	return selected_indices

def all_frame_overlap_sample(CLIP_LENGTH, start_point, end_point, global_min=0):
	offset = OVERLAP
	selected_indices = []

	for i in range(start_point, end_point+1, offset):
		if(i + CLIP_LENGTH) < (end_point+1):
			selected_indices.extend(range(i,i+CLIP_LENGTH))
		else:
			selected_indices.extend(range(end_point-CLIP_LENGTH+1,end_point+1))

	return selected_indices


def LRCN_classify_video(videos, labels, minima, maxima, net, transformer, is_flow):
	labels_prediction = []
	total_time = 0

	for video_counter, video in enumerate(videos):
		label = labels[video_counter]
		minimum = minima[video_counter]
		maximum = maxima[video_counter]
		
		print(str(video_counter) + ' ' + video + ' ' + str(label), end='')

		if is_flow == False:
			frames = glob.glob('%s%s/*.jpg' %(RGB_video_path, video))
		else:
			frames = glob.glob('%s%s/*.jpg' %(flow_video_path, video))
		frames.sort()

		vid_length = len(frames)

		if START_FRAME == '0':
			start_frame = 0
		if START_FRAME == 'gmin':
			start_frame = minimum
		if START_FRAME == 'lmax':
			start_frame = maximum

		if END_FRAME == '0':
			end_frame = 0
		if END_FRAME == 'gmin':
			end_frame = minimum
		if END_FRAME == 'lmax':
			end_frame = maximum
		if END_FRAME == 'vlength':
			end_frame = vid_length-1

		if OVERLAP == 0:
			# All frame sample
			selected_indices = all_frame_sample(CLIP_LENGTH, start_frame, end_frame)
		else:
			# All frame overlap sample
			selected_indices = all_frame_overlap_sample(CLIP_LENGTH, start_frame, end_frame)

		# print(selected_indices)

		selected_frames = []
		for i in selected_indices:
			selected_frame = frames[i]
			selected_frames.append(selected_frame)
		
		input_images = []
		for im in selected_frames:
			input_im = caffe.io.load_image(im)
			if(input_im.shape[0] < 240):
				input_im = caffe.io.resize_image(input_im, (240, 320))
			input_images.append(input_im)


		input_data = []
		input_data.extend(input_images)

		output_predictions = np.zeros((len(input_data), 101))

		video_time = 0
		for i in range(0, len(input_data), CLIP_LENGTH):
			clip_input = input_data[i:i+CLIP_LENGTH]
			clip_input = caffe.io.oversample(clip_input, [227, 227])

			caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,2,1]], dtype=np.float32)

			for ix, inputs in enumerate(clip_input):
				caffe_in[ix] = transformer.preprocess('data', inputs)

				# print(caffe_in.shape)

			if OVERLAP == 0:
				# Divide into blocks non-overlap
				clip_clip_markers = np.ones((CLIP_LENGTH*10, 1, 1, 1))
				if i == 0:					
					clip_clip_markers[0:10,:,:,:] = 0
			else:
				# Divide into 16-frame blocks with overlap
				clip_clip_markers = np.ones((CLIP_LENGTH*10, 1, 1, 1))
				clip_clip_markers[0:10,:,:,:] = 0	

			# print('clip_clip_markers: ' + str(clip_clip_markers.shape))	

			start = time.time()
			out = net.forward_all(data=caffe_in, clip_markers = np.array(clip_clip_markers))
			# print(out['probs'].shape)
			end = time.time()
			video_time += end-start


			output_predictions[i:i+CLIP_LENGTH] = np.mean(out['probs'], 1)


		for k in range(0, len(output_predictions)):
			frame_label = output_predictions[k].argmax()

		print(' ' + str(video_time), end = '')
		if video_counter != 0:
			total_time += video_time
		
		# Original
		output_predictions_final = np.mean(output_predictions, 0)
		
		f_out.writelines("%s " %video)
		f_out.writelines("%s " %label)
		f_out.writelines("%.6f " %prob for prob in output_predictions_final)
		f_out.writelines("\n")

		label_prediction = np.mean(output_predictions,0).argmax()
		print(' ' + str(label_prediction))
		labels_prediction.append(label_prediction)

		if(video_counter % 100 == 0 and video_counter != 0):
			print('Accuracy: ' + str(accuracy_score(labels[0:video_counter+1], labels_prediction)*100) + '% ================================')
			print('Average time: ' + str(total_time/(video_counter)) + ' second')

	print('Average time: ' + str(total_time/(len(videos)-1)) + ' second')
	print('Total time: ' + str(total_time) + ' second')

	return labels_prediction


#Models and weights
lstm_model = '../deploy_lstm.prototxt'
RGB_lstm = '../RGB_lstm_model_iter_30000.caffemodel'
flow_lstm = '../flow_lstm_model_iter_50000.caffemodel'

# Read test list 
f = open(TEST_FILE, 'r')
f_lines = f.readlines()
f.close()
videos = []
labels = []
minima = []
maxima = []
for _, line in enumerate(f_lines):
	line = line.split(' ')
	video = line[0] # video name
	label = int(line[1]) # video label
	minimum = int(line[2]) # global minimum
	maximum = int(line[3]) # adjacent maximum

	videos.append(video)
	labels.append(label)
	minima.append(minimum)
	maxima.append(maximum)

if MODALITY == 'rgb':
	# RGB net
	RGB_lstm_net =  caffe.Net(lstm_model, RGB_lstm, caffe.TEST)
	predicted_labels = LRCN_classify_video(videos, labels, minima, maxima, RGB_lstm_net, transformer_RGB, False)
	print('Accuracy: ' + str(accuracy_score(labels, predicted_labels)*100) + '%')
	del RGB_lstm_net

if MODALITY == 'flow':
	# Flow net
	flow_lstm_net =  caffe.Net(lstm_model, flow_lstm, caffe.TEST)
	predicted_labels = LRCN_classify_video(videos, labels, minima, maxima, flow_lstm_net, transformer_flow, True)
	print('Accuracy: ' + str(accuracy_score(labels, predicted_labels)*100) + '%')
	del flow_lstm_net


print('INFO -------------------------------------------------')
print('test file: ' + TEST_FILE)
print('out_file: ' + OUT_FILE)
print('clip_length: ' + str(CLIP_LENGTH))
print('modality: ' + MODALITY)
print('overlap: ' + str(OVERLAP))
print('starting from: ' + START_FRAME)
print('end at: ' + END_FRAME)
print('------------------------------------------------------')


f_out.close()
