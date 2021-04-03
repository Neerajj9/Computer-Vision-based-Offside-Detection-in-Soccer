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

	allTeamClassificationFeatures_upper = []
	allTeamClassificationFeatures_lower = []
	
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
				# print(ptsx, ptsy, ':', min_x, min_y)

				if ptsy > 0.0 and ptsx > 0.0:
					player_parts.update({names[j] : {'x' : ptsx , 'y' :ptsy} })

					x = x + 1
					if min_x > ptsx:
						min_x = ptsx
						min_y = ptsy

		# print('\n', min_x, min_y)				
		    
		teamClassificationFeatures_upper = []
		teamClassificationFeatures_lower = []
		
		allXCoords_upper = []
		allYCoords_upper = []
		allXCoords_lower = []
		allYCoords_lower = []

		leftUpperPointFound = False
		rightUpperPointFound = False
		leftMidPointFound = False
		rightMidPointFound = False
		leftLowerPointFound = False
		rightLowerPointFound = False
		
		if 'rightShoulder' in list(player_parts.keys()):
			allXCoords_upper.append(int(player_parts['rightShoulder']['x']))
			allYCoords_upper.append(int(player_parts['rightShoulder']['y']))
			rightUpperPointFound = True
		if 'leftShoulder' in list(player_parts.keys()):
			allXCoords_upper.append(int(player_parts['leftShoulder']['x']))
			allYCoords_upper.append(int(player_parts['leftShoulder']['y']))
			leftUpperPointFound = True
		
		if 'rightHip' in list(player_parts.keys()):
			allXCoords_upper.append(int(player_parts['rightHip']['x']))
			allYCoords_upper.append(int(player_parts['rightHip']['y']))
			allXCoords_lower.append(int(player_parts['rightHip']['x']))
			allYCoords_lower.append(int(player_parts['rightHip']['y']))
			rightMidpointFound = True
		if 'leftHip' in list(player_parts.keys()):
			allXCoords_upper.append(int(player_parts['leftHip']['x']))
			allYCoords_upper.append(int(player_parts['leftHip']['y']))
			allXCoords_lower.append(int(player_parts['leftHip']['x']))
			allYCoords_lower.append(int(player_parts['leftHip']['y']))
			leftMidPointFound = True

		if 'rightKnee' in list(player_parts.keys()):
			allXCoords_lower.append(int(player_parts['rightKnee']['x']))
			allYCoords_lower.append(int(player_parts['rightKnee']['y']))
			rightLowerPointFound = True		
		if 'leftKnee' in list(player_parts.keys()):
			allXCoords_lower.append(int(player_parts['leftKnee']['x']))
			allYCoords_lower.append(int(player_parts['leftKnee']['y']))
			leftLowerPointFound = True

		allXCoords_upper.sort()
		allYCoords_upper.sort()
		allXCoords_lower.sort()
		allYCoords_lower.sort()

		if (len(allXCoords_upper) < 3) or (len(allXCoords_lower) < 3):
			continue
			
		if mask[int(min_x) , int(min_y)] == 0:
			continue

		# Upper body features
		if len(allXCoords_upper) == 4:
			if 0 > allXCoords_upper[0]:
				allXCoords_upper[0] = 0
			if 0 > allXCoords_upper[3]:
				allXCoords_upper[3] = 0
			if 0 > allXCoords_upper[0] and 0 > allXCoords_upper[3]:
				allXCoords_upper[3] = 2
			if 0 > allYCoords_upper[0]:
				allYCoords_upper[0] = 0
			if 0 > allYCoords_upper[3]:
				allYCoords_upper[3] = 0
			if 0 > allYCoords_upper[0] and 0 > allYCoords_upper[3]:
				allYCoords_upper[3] = 2 
			allXCoords_upper.sort()
			allYCoords_upper.sort() 
			allColors = image2[allXCoords_upper[0]:allXCoords_upper[3], allYCoords_upper[0]:allYCoords_upper[3]]
			image = cv2.rectangle(image, (allYCoords_upper[0], allXCoords_upper[0]), (int(allYCoords_upper[3]-abs(allYCoords_upper[3]-allYCoords_upper[0])/2), allXCoords_upper[3]), (255, 0, 0), 1)
			chans = cv2.split(allColors)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeatures_upper.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeatures_upper = [float(i)/max(teamClassificationFeatures_upper) for i in teamClassificationFeatures_upper]
			allTeamClassificationFeatures_upper.append(teamClassificationFeatures_upper)

		if len(allXCoords_upper) == 3:
			if 0 > allXCoords_upper[0]:
				allXCoords_upper[0] = 0
			if 0 > allXCoords_upper[2]:
				allXCoords_upper[2] = 0
			if 0 > allXCoords_upper[0] and 0 > allXCoords_upper[2]:
				allXCoords_upper[2] = 10 
			if 0 > allYCoords_upper[0]:
				allYCoords_upper[0] = 0
			if 0 > allYCoords_upper[2]:
				allYCoords_upper[2] = 0
			if 0 > allYCoords_upper[0] and 0 > allYCoords_upper[2]:
				allYCoords_upper[2] = 10  
			allXCoords_upper.sort()
			allYCoords_upper.sort()
			allColors = image2[allXCoords_upper[0]:allXCoords_upper[2], allYCoords_upper[0]:allYCoords_upper[2]]
			image = cv2.rectangle(image, (allYCoords_upper[0], allXCoords_upper[0]), (int(allYCoords_upper[3]-abs(allYCoords_upper[2]-allYCoords_upper[0])/2), allXCoords_upper[2]), (255, 0, 0), 1)
			chans = cv2.split(allColors)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeatures_upper.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeatures_upper = [float(i)/max(teamClassificationFeatures_upper) for i in teamClassificationFeatures_upper]
			allTeamClassificationFeatures_upper.append(teamClassificationFeatures_upper)

		if len(allXCoords_upper) == 2:
			# if (leftUpperPointFound and rightMidPointFound) or (leftMidPointFound and rightUpperPointFound):
			if 0 > allXCoords_upper[0]:
				allXCoords_upper[0] = 0
			if 0 > allXCoords_upper[1]:
				allXCoords_upper[1] = 0
			if 0 > allXCoords_upper[0] and 0 > allXCoords_upper[1]:
				allXCoords_upper[1] = 2 
			if 0 > allYCoords_upper[0]:
				allYCoords_upper[0] = 0
			if 0 > allYCoords_upper[1]:
				allYCoords_upper[1] = 0
			if 0 > allYCoords_upper[0] and 0 > allYCoords_upper[1]:
				allYCoords_upper[1] = 2
			allXCoords_upper.sort()
			allYCoords_upper.sort()
			allColors = image2[allXCoords_upper[0]:allXCoords_upper[1], allYCoords_upper[0]:allYCoords_upper[1]]
			image = cv2.rectangle(image, (allYCoords_upper[0], allXCoords_upper[0]), (int(allYCoords_upper[1]-abs(allYCoords_upper[3]-allYCoords_upper[0])/2), allXCoords_upper[1]), (255, 0, 0), 1)
			'''reqInLinePoints = bresenham_march(image2, [allXCoords[0], allYCoords[0]], [allXCoords[1], allYCoords[1]])
			allColors = []
			for point in reqInLinePoints:
				allColors.append(point[1].reshape(1,3))
			reqInLinePoints = bresenham_march(image2, [allXCoords[0], allYCoords[1]], [allXCoords[1], allYCoords[0]])
			for point in reqInLinePoints:
				allColors.append(point[1].reshape(1,3))'''
			chans = cv2.split(allColors)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeatures_upper.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeatures_upper = [float(i)/max(teamClassificationFeatures_upper) for i in teamClassificationFeatures_upper]
			allTeamClassificationFeatures_upper.append(teamClassificationFeatures_upper)
		
		# Lower body features
		if len(allXCoords_lower) == 4:
			if 0 > allXCoords_lower[0]:
				allXCoords_lower[0] = 0
			if 0 > allXCoords_lower[3]:
				allXCoords_lower[3] = 0
			if 0 > allXCoords_lower[0] and 0 > allXCoords_lower[3]:
				allXCoords_lower[3] = 2
			if 0 > allYCoords_lower[0]:
				allYCoords_lower[0] = 0
			if 0 > allYCoords_lower[3]:
				allYCoords_lower[3] = 0
			if 0 > allYCoords_lower[0] and 0 > allYCoords_lower[3]:
				allYCoords_lower[3] = 2 
			allXCoords_lower.sort()
			allYCoords_lower.sort() 
			allColors = image2[allXCoords_lower[0]:allXCoords_lower[3], allYCoords_lower[0]:allYCoords_lower[3]]
			image = cv2.rectangle(image, (allYCoords_lower[0], allXCoords_lower[0]), (allYCoords_lower[3], allXCoords_lower[3]), (255, 0, 0), 1)
			chans = cv2.split(allColors)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeatures_lower.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeatures_lower = [float(i)/max(teamClassificationFeatures_lower) for i in teamClassificationFeatures_lower]
			allTeamClassificationFeatures_lower.append(teamClassificationFeatures_lower)

		if len(allXCoords_lower) == 3:
			if 0 > allXCoords_lower[0]:
				allXCoords_lower[0] = 0
			if 0 > allXCoords_lower[2]:
				allXCoords_lower[2] = 0
			if 0 > allXCoords_lower[0] and 0 > allXCoords_lower[2]:
				allXCoords_lower[2] = 10 
			if 0 > allYCoords_lower[0]:
				allYCoords_lower[0] = 0
			if 0 > allYCoords_lower[2]:
				allYCoords_lower[2] = 0
			if 0 > allYCoords_lower[0] and 0 > allYCoords_lower[2]:
				allYCoords_lower[2] = 10  
			allXCoords_lower.sort()
			allYCoords_lower.sort()
			allColors = image2[allXCoords_lower[0]:allXCoords_lower[2], allYCoords_lower[0]:allYCoords_lower[2]]
			image = cv2.rectangle(image, (allYCoords_lower[0], allXCoords_lower[0]), (allYCoords_lower[2], allXCoords_lower[2]), (255, 0, 0), 1)
			chans = cv2.split(allColors)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeatures_lower.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeatures_lower = [float(i)/max(teamClassificationFeatures_lower) for i in teamClassificationFeatures_lower]
			allTeamClassificationFeatures_lower.append(teamClassificationFeatures_lower)

		if len(allXCoords_lower) == 2:
			# if (leftMidPointFound and rightLowerPointFound) or (leftLowerPointFound and rightMidPointFound):
			if 0 > allXCoords_lower[0]:
				allXCoords_lower[0] = 0
			if 0 > allXCoords_lower[1]:
				allXCoords_lower[1] = 0
			if 0 > allXCoords_lower[0] and 0 > allXCoords_lower[1]:
				allXCoords_lower[1] = 2 
			if 0 > allYCoords_lower[0]:
				allYCoords_lower[0] = 0
			if 0 > allYCoords_lower[1]:
				allYCoords_lower[1] = 0
			if 0 > allYCoords_lower[0] and 0 > allYCoords_lower[1]:
				allYCoords_lower[1] = 2
			allXCoords_lower.sort()
			allYCoords_lower.sort()
			allColors = image2[allXCoords_lower[0]:allXCoords_lower[1], allYCoords_lower[0]:allYCoords_lower[1]]
			image = cv2.rectangle(image, (allYCoords_lower[0], allXCoords_lower[0]), (allYCoords_lower[1], allXCoords_lower[1]), (255, 0, 0), 1)
			'''reqInLinePoints = bresenham_march(image2, [allXCoords[0], allYCoords[0]], [allXCoords[1], allYCoords[1]])
			allColors = []
			for point in reqInLinePoints:
				allColors.append(point[1].reshape(1,3))
			reqInLinePoints = bresenham_march(image2, [allXCoords[0], allYCoords[1]], [allXCoords[1], allYCoords[0]])
			for point in reqInLinePoints:
				allColors.append(point[1].reshape(1,3))'''
			chans = cv2.split(allColors)
			colors = ("b", "g", "r")
			for (chan, color) in zip(chans, colors):
				hist = cv2.calcHist([chan], [0], None, [3], [0, 256])
				teamClassificationFeatures_lower.extend([np.int64(x[0]) for x in hist])
				teamClassificationFeatures_lower = [float(i)/max(teamClassificationFeatures_lower) for i in teamClassificationFeatures_lower]
			allTeamClassificationFeatures_lower.append(teamClassificationFeatures_lower)

		'''distance_keep = np.sqrt(np.sum(np.square(np.array(teamClassificationFeatures) - np.array(keeper))))
		if distance_keep < team_class_num:
			team_class_num = distance_keep
			keeper_id = i
		
		distance_ref = np.sqrt(np.sum(np.square(np.array(teamClassificationFeatures) - np.array(referee))))
		if distance_ref < ref_num:
			ref_ids.append(i)'''
		
		temp = None
		
		all_info.append([i, temp, player_parts, [min_x,min_y]])
	
	teamClassifier_upper = KMeans(n_clusters = 2)
	teamLabels_upper = teamClassifier_upper.fit_predict(allTeamClassificationFeatures_upper)
	teamClassifierDBSCAN_upper = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(allTeamClassificationFeatures_upper)
	teamLabelsDBSCAN_upper = teamClassifierDBSCAN_upper.labels_

	teamClassifier_lower = KMeans(n_clusters = 2)
	teamLabels_lower = teamClassifier_lower.fit_predict(allTeamClassificationFeatures_lower)
	teamClassifierDBSCAN_lower = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(allTeamClassificationFeatures_lower)
	teamLabelsDBSCAN_lower = teamClassifierDBSCAN_lower.labels_

	isKeeperFound = False
	isRefFound = False

	# Mapping clusters
	# Create cost matrix
	CM = np.zeros(shape=(2, 2))
	for i in range(2):
		for j in range(2):
			CM[i][j] =  calc_dist_between_points(teamClassifier_upper.cluster_centers_[i], \
				teamClassifier_lower.cluster_centers_[j], \
				len(teamClassifier_lower.cluster_centers_))

	# Hungarian algorithm
	row_ind, col_ind = linear_sum_assignment(CM)
	upper_to_lower_map = dict(zip(row_ind, col_ind))
	lower_to_upper_map = dict(zip(col_ind, row_ind))
		
	for player_itr in range(len(all_info)):

		dist_from_centroid_upper = np.sqrt(np.sum(np.square(np.array(allTeamClassificationFeatures_upper[player_itr]) - \
			np.array(teamClassifier_upper.cluster_centers_[teamLabels_upper[player_itr]]))))
		dist_from_centroid_lower = np.sqrt(np.sum(np.square(np.array(allTeamClassificationFeatures_lower[player_itr]) - \
			np.array(teamClassifier_lower.cluster_centers_[teamLabels_lower[player_itr]]))))

		if (teamLabelsDBSCAN_upper[player_itr] == -1) or (teamLabelsDBSCAN_lower[player_itr] == -1):
			all_info[player_itr][1] = -1
		else:
			all_info[player_itr][1] = teamLabels_upper[player_itr] * 10 + teamLabels_lower[player_itr]
		'''elif (upper_to_lower_map[teamLabels_upper[player_itr]] == teamLabels_lower[player_itr]):
			all_info[player_itr][1] = teamLabels_upper[player_itr]
		else:
			if dist_from_centroid_upper < dist_from_centroid_lower:
				all_info[player_itr][1] = teamLabels_upper[player_itr]
			else:
				all_info[player_itr][1] = lower_to_upper_map[teamLabels_lower[player_itr]]'''
		
		'''if all_info[player_itr][0] == keeper_id:
			isKeeperFound = True
			all_info[player_itr][1] = 2'''

		'''if all_info[player_itr][1] in ref_ids:
			isRefFound = True
			all_info[player_itr][1] = 3'''
			
	print("ref:", ref_ids,"keeper:", keeper_id, "score", team_class_num)	
		
	return all_info, isKeeperFound, isRefFound, image