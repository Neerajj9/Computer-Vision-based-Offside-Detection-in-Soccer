from PlayerSegmentationUtils import *
from VanishingPointUtils import *
import sys
#sys.path.append('/posenet_python')
import matplotlib.pyplot as plt
#from posenet_python.posenet import *
#from posenet_python.image_demo import *

parts = ['rightShoulder', 'leftShoulder', 'leftHip', 'rightHip', 'rightKnee', 'leftKnee', 'rightAnkle', 'leftAnkle']

# get pose estimations on boxes
def get_pose_estimations(image):
	
	contours, mask = get_contours(image)
	boxes, scores = get_boxes(image, contours)
	final_boxes = non_max(boxes, np.array(scores), 0.1)
	
	ans, teamLabels = get_info(final_boxes, image)
	# print(teamLabels)

	pose_estimations = []

	for values in ans:
		for value in values:
			for pose_estimation in value:
				pose_estimations.append(pose_estimation)

	for pose_itr in range(len(pose_estimations)):
	    font = cv2.FONT_HERSHEY_SIMPLEX
	    cv2.putText(image, str(pose_estimations[pose_itr][0])+':'+str(teamLabels[pose_itr]), (pose_estimations[pose_itr][-1][1], pose_estimations[pose_itr][-1][0]) , font, 1, (200,255,155), 2, cv2.LINE_AA)
	plt.figure(figsize=(20,20))
	plt.imshow(image)

	return pose_estimations

# updates leftmost point according to the angle at Vanishing Point instead of 'X' co-ordinate
def update_pose_left_most_point(vertical_vanishing_point, horizontal_vanishing_point, pose_estimations, image, goalDirection):


	plane = get_ground_plane(image, [vertical_vanishing_point[1], vertical_vanishing_point[0]], [horizontal_vanishing_point[1], horizontal_vanishing_point[0]], 'right')


	for pose in pose_estimations:
	    currLeftMostKey = ''
	    currKeyPointAngles = []
	    currLeftmostPt = pose[-1]
	    currLeftmostPtAngle = get_angle(vertical_vanishing_point, (currLeftmostPt[1], currLeftmostPt[0]), image, goalDirection)
	    for keyPoint in pose[2]:
	        currKeyPtAngle = get_angle(vertical_vanishing_point,(pose[2][keyPoint]['y'], pose[2][keyPoint]['x']), image, goalDirection)
	        if currKeyPtAngle < currLeftmostPtAngle:
	            currLeftmostPtAngle = currKeyPtAngle
	            currLeftmostPt = [pose[2][keyPoint]['x'], pose[2][keyPoint]['y']]
	            currLeftMostKey = keyPoint
	    
	    if 'Ankle' in currLeftMostKey:
	    	pose[-1] = currLeftmostPt
	    elif currLeftMostKey == '':
	    	point = [pose[-1][1], pose[-1][0]]
	    	new_point = project_point_on_plane(plane, point , pose , 2.67 , image)
    		pose[-1] = new_point
    		cv2.line(image , (int(point[0]) , int(point[1])) , (int(new_point[1]) , int(new_point[0])) , (255,0,0) , 2 )
	    else:
	    	point = [pose[2][currLeftMostKey]['y'], pose[2][currLeftMostKey]['x']]
    		new_point = project_point_on_plane(plane, point , pose , 2.67 , image)
    		pose[-1] = new_point
    		cv2.line(image , (int(point[0]) , int(point[1])) , (int(new_point[1]) , int(new_point[0])) , (255,0,0) , 2 )
	    	
	return pose_estimations

# adds the angle subtended at vanishing point
def get_leftmost_point_angles(vertical_vanishing_point, pose_estimations, image, goalDirection):

	for pose in pose_estimations:
	    curr_angle = get_angle(vertical_vanishing_point, (pose[-1][-1], pose[-1][0]), image, goalDirection) 
	    pose.append(curr_angle)

	return pose_estimations