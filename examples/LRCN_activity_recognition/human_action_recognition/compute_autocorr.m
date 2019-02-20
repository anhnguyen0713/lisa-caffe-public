frames_folder = 'D:\Documents\Deep Learning\Motion recognition\frames\';
file_name = 'D:\Documents\Deep Learning\Motion recognition\Sampling-in-test\ucf101_split1_testVideos.txt';
f = fopen(file_name, 'r');

f_out = fopen('video_corr.txt', 'w');

videos = {};

line = fgetl(f);
while ischar(line)
   C = strsplit(line, ' ');
   label = C{2};
   V = strsplit(C{1}, '/');
   video = V{2};
   
   video_folder = [frames_folder video];
   if 7~=exist(video_folder,'dir')
       disp(['Missing ' video]);
   else
      disp(['Processing ' video]);
      frames = dir([video_folder '\*.jpg']);
      num_frames = size(frames, 1);
      
      series_RGB = [];
      for i=1:num_frames
          img = imread([video_folder '\' frames(i).name]);
          mean_R = mean(mean(img(:,:,1)));
%           mean_G = mean(mean(img(:,:,2)));
%           mean_B = mean(mean(img(:,:,3)));
%           mean_val = (mean_R + mean_G + mean_B)/3;
          series_RGB(i) = mean_R;
      end
      
      series_RGB_norm = series_RGB - mean(series_RGB);
      corr = xcorr(series_RGB_norm);
      corr_norm = corr - mean(corr);
      
      fprintf(f_out, '%s ', video);
      fprintf(f_out, '%s ', label);
      for i=num_frames:length(corr_norm)
          fprintf(f_out, '%.6f ', corr_norm(i));
      end
      fprintf(f_out, '\n');
   end
   
   line = fgetl(f);
end
fclose(f);
fclose(f_out);