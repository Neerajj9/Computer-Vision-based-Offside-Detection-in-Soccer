def get_offside_decision(pose_estimations, vertical_vanishing_point, attackingTeamId, defendingTeamId, isKeeperFound):

	# get last defending man
	currMinAngle = 360.0
	last_defending_man = -1
	for pose in pose_estimations:
		if pose[1] in [defendingTeamId, 2]:
			if pose[-1] not in ['on', 'off', 'def', 'ref']:
				if pose[-1] < currMinAngle:
					currMinAngle = pose[-1]
					last_defending_man = pose[0]
			elif pose[-2] < currMinAngle:
				currMinAngle = pose[-2]
				last_defending_man = pose[0]
	exclude_last_man_id = last_defending_man
	currMinAngle = 360.0
	last_defending_man = -1
	for pose in pose_estimations:
		if (pose[1] == defendingTeamId or pose[1] == 2) and pose[0] != exclude_last_man_id:
			if pose[-1] not in ['on', 'off', 'def', 'ref']:
				if pose[-1] < currMinAngle:
					currMinAngle = pose[-1]
					last_defending_man = pose[0]
			elif pose[-2] < currMinAngle:
				currMinAngle = pose[-2]
				last_defending_man = pose[0]
	# get decision for each detected player
	for pose in pose_estimations:
		# attacking team
		if pose[1] == attackingTeamId:
			if pose[-1] not in ['on', 'off', 'def', 'ref']:
				if pose[-1] < currMinAngle:
					pose.append('off')
				else:
					pose.append('on')
			else:
				if pose[-2] < currMinAngle:
					pose[-1] = 'off'
				else:
					pose[-1] = 'on'
		# defending team, append 'def' to maintain uniformity in data structure
		else:
			if pose[-1] not in ['on', 'off', 'def', 'ref']:
				if pose[1] == 3:
					pose.append('ref')
				else:
					pose.append('def')
			else:
				pose[-1] = 'def'

	return pose_estimations, last_defending_man
