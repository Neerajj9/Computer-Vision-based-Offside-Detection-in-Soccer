import numpy as np

# TODO : add code for referee
def get_team_classifications(teamColor1, teamColor2, refColor, keeper1Color, keeper2Color, pose_estimations):

	for pose in pose_estimations:
    
	    if len(pose[1]) < 2:
	        
	        pose.append('color not found')
	        continue
	      
	    colorDiffs = {}
	    colorList = np.array(pose[1][0]) + np.array(pose[1][1])
	    colorList = np.divide(colorList, 2)
	    colorList = colorList.tolist()
	    diffTeam1 = list(abs(np.array(teamColor1) - np.array(colorList)))
	    colorDiffs['team1'] = diffTeam1
	    diffTeam2 = list(abs(np.array(teamColor2) - np.array(colorList))) 
	    colorDiffs['team2'] = diffTeam2
	    diffRef = list(abs(np.array(refColor) - np.array(colorList)))
	    colorDiffs['ref'] = diffRef
	    diffKeep1 = list(abs(np.array(refColor) - np.array(colorList)))
	    colorDiffs['keep1'] = diffKeep1
	    diffKeep2 = list(abs(np.array(refColor) - np.array(colorList)))
	    colorDiffs['keep2'] = diffKeep2

	    for key in colorDiffs.keys():

	    	colorDiffs[key] = sum(colorDiffs[key]) / len(colorDiffs[key])

	    colorDiffs = {k: v for k, v in sorted(colorDiffs.items(), key=lambda item: item[1])}

	    for key in colorDiffs.keys(): 
	    	pose.append(key)
	    	break

	return pose_estimations