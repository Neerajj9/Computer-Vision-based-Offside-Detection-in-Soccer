import os
import sys
import cv2
import math
import numpy as np
sys.path.append(os.path.dirname(__file__) + "/../")
from PlayerSegmentationUtils import *
from imageio import imread, imsave
from util.config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")


cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

def calc_dist_between_points(point1, point2, dimensions):
	sum = 0
	for n in range(dimensions):
		sum += (point1[n] - point2[n]) ** 2
	return math.sqrt(sum)

def check_points(allPoints, centroid):
	checked_points = []
	outliers = []
	avg_dist = 0
	sum_dist_from_centroid = 0
	for n in range(len(allPoints)):
		sum_dist_from_centroid = calc_dist_between_points(allPoints[n], centroid, len(centroid))
		checked_points.append(sum_dist_from_centroid)
			
	checked_points.sort()
	q1 , q3 = np.percentile(checked_points , [20,80])
	iqr = q1 - q3
	lower_range = q1 - (1.5 * iqr)
	upper_range = q3 + (1.5 * iqr)
	ans = []

	for n in range(len(allPoints)):
		if checked_points[n] < upper_range or checked_points[n] > lower_range:
			ans.append(n)

	return ans

def bresenham_march(img, p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    #tests if any coordinate is outside the image
    if ( 
        x1 >= img.shape[0]
        or x2 >= img.shape[0]
        or y1 >= img.shape[1]
        or y2 >= img.shape[1]
    ): #tests if line is in image, necessary because some part of the line must be inside, it respects the case that the two points are outside
        if not cv2.clipLine((0, 0, *img.shape), p1, p2):
            print("not in region")
            return

    steep = math.fabs(y2 - y1) > math.fabs(x2 - x1)
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # takes left to right
    also_steep = x1 > x2
    if also_steep:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = x2 - x1
    dy = math.fabs(y2 - y1)
    error = 0.0
    delta_error = 0.0
    # Default if dx is zero
    if dx != 0:
        delta_error = math.fabs(dy / dx)

    y_step = 1 if y1 < y2 else -1

    y = y1
    ret = []
    for x in range(x1, x2):
        p = (y, x) if steep else (x, y)
        if p[0] < img.shape[0] and p[1] < img.shape[1]:
            ret.append((p, img[p]))
        error += delta_error
        if error >= 0.5:
            y += y_step
            error -= 1
    if also_steep:  # because we took the left to right instead
        ret.reverse()
    return ret

def return_pose(image , image2 , keeper , referee):
	
	keeper_id = -1
	ref_ids= []
	
	team_class_num = 0.25 # sys.maxsize
	ref_num = 0.10
	
	contours, mask = get_contours(image)
		
	all_info = []

	image_batch = data_to_input(image)

	# Compute prediction with the CNN
	outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
	scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

	detections = extract_detections(cfg, scmap, locref, pairwise_diff)
	unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
	person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

	cl = [(255,0,0),(0,0,255),(0,255,0),(255,255,0),(0,255,255),(255,0,255),(0,0,0)]

	names = {6:'rightShoulder', 5:'leftShoulder', 11:'leftHip', 12:'rightHip',14:'rightKnee',13:'leftKnee',16:'rightAnkle',15:'leftAnkle'}	

	allTeamClassificationFeatures = []
	allTeamClassificationFeaturesDBSCAN = []
	
	# TODO - Use mid-back to knee features
	for i in range(len(person_conf_multi)):
		
		x = 0
		player_parts = {}
		min_x = image2.shape[0] - 1
		min_y = image2.shape[1] - 1

		for j in range(17):
			
			printFlag = True
			if j in [5, 6, 15, 16, 11, 12, 13, 14]:
				ptsy, ptsx = person_conf_multi[i,j,:]

				if ptsy > 0.0 and ptsx > 0.0:
					player_parts.update({names[j] : {'x' : ptsx , 'y' :ptsy} })
					x = x + 1
					
					if min_x > ptsx:
						min_x = ptsx
						min_y = ptsy
		    
		teamClassificationFeatures = []
		teamClassificationFeaturesDBSCAN = []
		
		allXCoords = []
		allYCoords = []

		leftUpperPointFound = False
		rightUpperPointFound = False
		leftLowerPointFound = False
		rightLowerPointFound = False
		hipPointsFound = False
		
		if 'rightShoulder' in list(player_parts.keys()):
			allXCoords.append(int(player_parts['rightShoulder']['x']))
			allYCoords.append(int(player_parts['rightShoulder']['y']))
			rightUpperPointFound = True
		if 'leftShoulder' in list(player_parts.keys()):
			allXCoords.append(int(player_parts['leftShoulder']['x']))
			allYCoords.append(int(player_parts['leftShoulder']['y']))
			leftUpperPointFound = True

		if 'rightKnee' in list(player_parts.keys()):
			allXCoords.append(int(player_parts['rightKnee']['x']))
			allYCoords.append(int(player_parts['rightKnee']['y']))
			rightLowerPointFound = True		
		if 'leftKnee' in list(player_parts.keys()):
			allXCoords.append(int(player_parts['leftKnee']['x']))
			allYCoords.append(int(player_parts['leftKnee']['y']))
			leftLowerPointFound = True

		allXCoords.sort()
		allYCoords.sort()
			
		if len(allXCoords) < 3:
			continue

		hipPointsFound = 'rightHip' in list(player_parts.keys()) and 'leftHip' in list(player_parts.keys())
		if hipPointsFound:
			hipPointsFound &= abs(int(player_parts['leftHip']['y']) - int(player_parts['rightHip']['y'])) > 5
		if hipPointsFound:
			allHipXCoords = []
			allHipYCoords = []
			allHipXCoords.append(int(player_parts['rightHip']['x']))
			allHipYCoords.append(int(player_parts['rightHip']['y']))
			allHipXCoords.append(int(player_parts['leftHip']['x'])) 
			allHipYCoords.append(int(player_parts['leftHip']['y']))
			allHipXCoords.sort()
			allHipYCoords.sort()
		
		if mask[int(min_x) , int(min_y)] == 0:
			continue

		if len(allXCoords) == 4:
			if 0 > allXCoords[0]:
				allXCoords[0] = 0
			if 0 > allXCoords[3]:
				allXCoords[3] = 0
			if 0 > allXCoords[0] and 0 > allXCoords[3]:
				allXCoords[3] = 2
			if 0 > allYCoords[0]:
				allYCoords[0] = 0
			if 0 > allYCoords[3]:
				allYCoords[3] = 0
			if 0 > allYCoords[0] and 0 > allYCoords[3]:
				allYCoords[3] = 2 
			allXCoords.sort()
			allYCoords.sort() 
			newXLower = int(allXCoords[0]+abs(allXCoords[3]-allXCoords[0])*0.4)
			newXUpper = int(allXCoords[3]-abs(allXCoords[3]-allXCoords[0])*0.2)
			if hipPointsFound:
				newYLower = int(allHipYCoords[0])
				newYUpper = int(allHipYCoords[-1])
			else:
				newYLower = int(allYCoords[0])
				newYUpper = int(allYCoords[3]) 
			allColors = image2[newXLower:newXUpper, newYLower:newYUpper]
			allColorsDBSCAN = image2[allXCoords[0]:allXCoords[-1], allYCoords[0]:allYCoords[-1]]
			image = cv2.rectangle(image, (newYLower, newXLower), (newYUpper, newXUpper), (255, 0, 0), 1)
			chans = cv2.split(allColors)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [64], [0, 256])
				teamClassificationFeatures.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeatures = [float(i)/max(teamClassificationFeatures) for i in teamClassificationFeatures]
			allTeamClassificationFeatures.append(teamClassificationFeatures)
			chans = cv2.split(allColorsDBSCAN)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeaturesDBSCAN.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeaturesDBSCAN = [float(i)/max(teamClassificationFeaturesDBSCAN) for i in teamClassificationFeaturesDBSCAN]
			allTeamClassificationFeaturesDBSCAN.append(teamClassificationFeaturesDBSCAN)

		if len(allXCoords) == 3:
			if 0 > allXCoords[0]:
				allXCoords[0] = 0
			if 0 > allXCoords[2]:
				allXCoords[2] = 0
			if 0 > allXCoords[0] and 0 > allXCoords[2]:
				allXCoords[2] = 10 
			if 0 > allYCoords[0]:
				allYCoords[0] = 0
			if 0 > allYCoords[2]:
				allYCoords[2] = 0
			if 0 > allYCoords[0] and 0 > allYCoords[2]:
				allYCoords[2] = 10  
			allXCoords.sort()
			allYCoords.sort()
			newXLower = int(allXCoords[0]+abs(allXCoords[2]-allXCoords[0])*0.4)
			newXUpper = int(allXCoords[2]-abs(allXCoords[2]-allXCoords[0])*0.2)
			if hipPointsFound:
				newYLower = int(allHipYCoords[0])
				newYUpper = int(allHipYCoords[-1])
			else:
				newYLower = int(allYCoords[0])
				newYUpper = int(allYCoords[2])
			allColors = image2[newXLower:newXUpper, newYLower:newYUpper]
			allColorsDBSCAN = image2[allXCoords[0]:allXCoords[-1], allYCoords[0]:allYCoords[-1]]
			image = cv2.rectangle(image, (newYLower, newXLower), (newYUpper, newXUpper), (255, 0, 0), 1)
			chans = cv2.split(allColors)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [64], [0, 256])
				teamClassificationFeatures.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeatures = [float(i)/max(teamClassificationFeatures) for i in teamClassificationFeatures]
			allTeamClassificationFeatures.append(teamClassificationFeatures)
			chans = cv2.split(allColorsDBSCAN)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeaturesDBSCAN.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeaturesDBSCAN = [float(i)/max(teamClassificationFeaturesDBSCAN) for i in teamClassificationFeaturesDBSCAN]
			allTeamClassificationFeaturesDBSCAN.append(teamClassificationFeaturesDBSCAN)

		if len(allXCoords) == 2:
			# if (leftUpperPointFound and rightLowerPointFound) or (leftLowerPointFound and rightUpperPointFound):
			if 0 > allXCoords[0]:
				allXCoords[0] = 0
			if 0 > allXCoords[1]:
				allXCoords[1] = 0
			if 0 > allXCoords[0] and 0 > allXCoords[1]:
				allXCoords[1] = 2 
			if 0 > allYCoords[0]:
				allYCoords[0] = 0
			if 0 > allYCoords[1]:
				allYCoords[1] = 0
			if 0 > allYCoords[0] and 0 > allYCoords[1]:
				allYCoords[1] = 2
			allXCoords.sort()
			allYCoords.sort()
			newXLower = int(allXCoords[0]+abs(allXCoords[1]-allXCoords[0])*0.35)
			newXUpper = int(allXCoords[1]*0.9) 
			allColors = image2[newXLower:newXUpper, allYCoords[0]:allYCoords[1]]
			allColorsDBSCAN = image2[allXCoords[0]:allXCoords[1], allYCoords[0]:allYCoords[1]]
			image = cv2.rectangle(image, (allYCoords[0], int(allXCoords[0]+abs(allXCoords[1]-allXCoords[0])*0.35)), (allYCoords[1], allXCoords[1]), (255, 0, 0), 1)
			chans = cv2.split(allColors)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeatures.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeatures = [float(i)/max(teamClassificationFeatures) for i in teamClassificationFeatures]
			allTeamClassificationFeatures.append(teamClassificationFeatures)
			chans = cv2.split(allColorsDBSCAN)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeaturesDBSCAN.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeaturesDBSCAN = [float(i)/max(teamClassificationFeaturesDBSCAN) for i in teamClassificationFeaturesDBSCAN]
			allTeamClassificationFeaturesDBSCAN.append(teamClassificationFeaturesDBSCAN)
		
		temp = None
		
		all_info.append([i, temp, player_parts, [min_x,min_y]])
	
	teamClassifier = KMeans(n_clusters = 2)
	teamLabels = teamClassifier.fit_predict(allTeamClassificationFeatures)
	teamClassifierDBSCAN = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(allTeamClassificationFeaturesDBSCAN)
	teamLabelsDBSCAN = teamClassifierDBSCAN.labels_

	isKeeperFound = False
	isRefFound = False

	dist_from_centroids = []
	for player_itr in range(len(all_info)):

		dist_from_centroid = np.sqrt(np.sum(np.square(np.array(allTeamClassificationFeatures[player_itr]) - \
			np.array(teamClassifier.cluster_centers_[teamLabels[player_itr]]))))
		dist_from_centroids.append(dist_from_centroid)

	normed_dist_from_centroids = [float(i)/sum(dist_from_centroids) for i in dist_from_centroids]
	print('normes dist', normed_dist_from_centroids)
		
	for player_itr in range(len(all_info)):

		if (teamLabelsDBSCAN[player_itr] == -1):
			all_info[player_itr][1] = -1
		else:
			all_info[player_itr][1] = teamLabels[player_itr]
			
	print("ref:", ref_ids,"keeper:", keeper_id, "score", team_class_num)	
		
	return all_info, isKeeperFound, isRefFound, image
