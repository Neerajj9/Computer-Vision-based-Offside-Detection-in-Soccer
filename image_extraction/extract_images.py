import copy
import cv2
import os
import sys

sys.path.append('..')
from vanishing_point_utils_for_images_scraping import get_vertical_lines

video_dir = 'videos/'
video_files = os.listdir(video_dir)

for video_file in video_files:

	cam = cv2.VideoCapture(video_dir+video_file)

	current_frame = 0

	while(True):

		ret, frame = cam.read()

		if ret:

			if current_frame % 2 == 0:

				temp_frame = copy.copy(frame) 

				lines = get_vertical_lines(temp_frame)
				name = video_file+'_'+str(current_frame)+'.jpg'
				print('Creating '+name, len(lines))
				
				if len(lines) > 1:

					cv2.imwrite(name, frame)
					print('Created '+name)

			current_frame += 1

		else:
			break

	cam.release()
