import os
import numpy as np
os.environ["GLOG_minloglevel"] = '2'
caffe_root = '../../../'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import scipy.io as sio
import glob
from sklearn.metrics import accuracy_score
from PIL import Image
import cv2
USE_GPU = True
if USE_GPU:
	caffe.set_device(0)
	caffe.set_mode_gpu()
else:
	caffe.set_mode_cpu()

# Some params
CLIP_LENGTH = 8
EPS = 1
ALPHA = 1
NUM_ITER = 0
ATTACK_RATE = 3
VIDEO_LIST = "ucf101_split1_testVideos_minima_maxima.txt"
VIDEO_PATH = "/media/anhnguyen/4C35-FA26/UCF101-split1"
AE_PATH = "/home/anhnguyen/action_recog/UCF-101/lrcn_fgsm_iter_" + str(EPS) + "_rate_" + str(ATTACK_RATE)
NET_PROTOTXT = "/home/anhnguyen/action_recog/lisa-caffe-public/examples/LRCN_activity_recognition/deploy_lstm_ae.prototxt"
WEIGHTS_PATH = "/home/anhnguyen/action_recog/lisa-caffe-public/examples/LRCN_activity_recognition/RGB_lstm_model_iter_30000.caffemodel"
OUTPUT = "results/rgb_center_fgsm_iter_" + str(EPS) + "_rate_" + str(ATTACK_RATE) + "_begin.txt"

f_out = open(OUTPUT,'w')

# Loading a network
net = caffe.Net(NET_PROTOTXT, WEIGHTS_PATH, caffe.TEST)

# # Display network layers
# print("Network layers:")
# for name, layer in zip(net._layer_names, net.layers):
# 	print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))

# # Display network blobs
# print("Blobs:")
# for name, blob in net.blobs.iteritems():
#     print("{:<5}:  {}".format(name, blob.data.shape))

# Display information
print("EPS = " + str(EPS))
print("ATTACK_RATE = " + str(ATTACK_RATE))
print("CLIP_LENGTH = " + str(CLIP_LENGTH))
print("VIDEO_LIST = " + VIDEO_LIST)
print("OUTPUT = " + OUTPUT)
print("AE_PATH = " + AE_PATH)
print("------------------------------------------")

#Initialize transformers 
def initialize_transformer(image_mean, is_flow):
	shape = (CLIP_LENGTH, 3, 227, 227)
	transformer = caffe.io.Transformer({'data': shape})
	channel_mean = np.zeros((3,227,227))
	for channel_index, mean_val in enumerate(image_mean):
		channel_mean[channel_index, ...] = mean_val
	transformer.set_mean('data', channel_mean)
	transformer.set_raw_scale('data', 255.0)
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

# Read video list
f = open(VIDEO_LIST, 'r')
f_lines = f.readlines()
f.close()

true_labels = []
pred_labels = []

for i_line, line in enumerate(f_lines):
	# Extract video name and its label in each line
	video = line.split(' ')[0]
	label = int(line.split(' ')[1])

	true_labels.append(label)

	if i_line==0:
		# Create output directory
		if(os.path.isdir(AE_PATH) == False):
			os.system("mkdir " + AE_PATH)

		out_dir = AE_PATH + "/" + video
		if(os.path.isdir(out_dir) == False):
			os.system("mkdir " + out_dir)

	# Get all frames in video directory
	vid_dir = VIDEO_PATH + "/" + video
	frames = glob.glob(vid_dir+"/*.jpg")
	frames.sort()
	num_frames = len(frames)

	# If number of frames is not multiplication of CLIP_LENGTH, duplicate the last frame
	while(len(frames) % CLIP_LENGTH != 0):
		frames.append(frames[-1])

	# Attack rate
	frames_to_attack = frames[0:len(frames):ATTACK_RATE]
	
	# Create array to store output probabilities
	probs = np.zeros((len(frames),101))

	# Divide into clips without overlap
	for i in range(0, len(frames), CLIP_LENGTH):
		# Read image data and preprocess image
		read_frames = []
		clip_data = []

		selected_frames = []

		for j in range(i,i+CLIP_LENGTH):
			selected_frames.append(frames[j])
			frame_data = caffe.io.load_image(frames[j])
			if(frame_data.shape[0] < 240):
				frame_data = caffe.io.resize_image(frame_data, (240,320))
			read_frames.append(frame_data)

		# print(read_frames)
		clip_data = caffe.io.oversample(read_frames, [227, 227])
		clip_data = np.array(list(clip_data[id_crop] for id_crop in [4,14,24,34,44,54,64,74]))

		# # Display input crops
		# for n in range(0,8):
		# 	im = clip_data[n,:,:,:]
		# 	ax = plt.subplot(1,8,n+1)
		# 	ax.imshow(im)
		# plt.show()
		# sys.exit()
		
		data_in = np.zeros(np.array(clip_data.shape)[[0,3,2,1]], dtype=np.float32)

		for ix, im in enumerate(clip_data):
			data_in[ix] = transformer_RGB.preprocess('data', im)

		# Generate clip markers, head of clip is 0, otherwise 1
		clip_markers_in = np.ones((CLIP_LENGTH,1,1,1))
		if(i==0):
			clip_markers_in[0,:,:,:] = 0

		# Generate label array
		label_in = np.zeros((CLIP_LENGTH,1,1,1), dtype=np.uint8)
		label_in[:] = label

		clip_min = data_in - EPS
		clip_max = data_in + EPS

		if NUM_ITER <= 0:
			num_iters = np.min([EPS + 4, 1.25*EPS])
			num_iters = int(np.max([np.ceil(num_iters), 1]))
		else:
			num_iters = NUM_ITER
			
		ae_crops = np.array(data_in)
		# Iterate to craft ae
		for k in range(num_iters):
			# Forward data
			net.blobs['data'].data[...] = ae_crops
			net.blobs['label'].data[...] = label_in
			net.blobs['clip_markers'].data[...] = clip_markers_in
			res = net.forward()

			# Calculate backward gradient of loss layer w.r.t input
			data_diff = net.backward(diffs=['data'])
			grad_data = data_diff['data']

			# Get sign of gradient
			signed_grad = np.sign(grad_data)

			# Step perturbation
			step_perturb = signed_grad * ALPHA

			# In each step add only small perturb
			for ix_sel, selected_frame in enumerate(selected_frames):
				if selected_frame in frames_to_attack:
					ae_crops[ix_sel,:,:,:] = np.clip(ae_crops[ix_sel,:,:,:] + step_perturb[ix_sel,:,:,:], clip_min[ix_sel,:,:,:], clip_max[ix_sel,:,:,:])

		# Predict ae crops
		net.blobs['data'].data[...] = ae_crops

		# Forward ae crops and predict labels
		res = net.forward()
		probs[i:i+CLIP_LENGTH] = np.mean(net.blobs['probs'].data,1)

		# Reverse effects of transformer perprocess
		ae_crops_deprocess = []
		for m in range(0,CLIP_LENGTH):
			ae_frame = ae_crops[m,:,:,:]
			ae_frame += ucf_mean_RGB
			ae_frame /= 255.0
			ae_frame = ae_frame[(2,1,0),:,:]
			ae_frame = ae_frame.transpose((1,2,0))
			
			# Clip exceeded values
			ae_frame[ae_frame > 1] = 1.
			ae_frame[ae_frame < 0] = 0.

			ae_crops_deprocess.append(ae_frame)

		ae_crops_deprocess = np.array(ae_crops_deprocess)
		
		# Replace center crops of raw images by ae crops
		ae_frames = np.array(read_frames)
		ae_frames[0:8,6:233,46:273,:] = ae_crops_deprocess[:,:,:,:]

		# Save ae frames
		if i_line == 0:
			for idx, frame in enumerate(ae_frames):
				img = Image.fromarray((frame*255).astype('uint8')) #convert image to uint8
				img.save(out_dir + "/" + selected_frames[idx].split('/')[-1][0:-4] + '_ae.jpg')

	# Calculate predicted result
	probs = probs.reshape(len(frames), 101)
	vid_probs = np.mean(probs,0)
	label_pred = vid_probs.argmax()
	pred_labels.append(label_pred)
	print(str(i_line)+" "+video+" "+str(label)+" "+str(label_pred))

	# Calculate partial accuracy
	if(i_line % 10 == 0 and i_line > 0):
		print('Accuracy: ' + str(accuracy_score(true_labels[0:i_line], pred_labels[0:i_line])*100) + '%')

	# Write video probabilities to file
	f_out.writelines("%s " %video)
	f_out.writelines("%s " %label)
	f_out.writelines("%.6f " %prob for prob in vid_probs)
	f_out.writelines("\n")
	# sys.exit()

print('Accuracy: ' + str(accuracy_score(true_labels, pred_labels)*100) + '%')
f_out.close()