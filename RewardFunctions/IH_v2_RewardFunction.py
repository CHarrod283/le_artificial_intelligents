import math

def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    speed = params['speed']
    
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']

    progress = params['progress']
    offtrack = params['is_offtrack']
    left = params['is_left_of_center']
    
    if offtrack:
        reward = 1e-3
    else
        reward = progress
    
        # Calculate 3 markers that are at varying distances away from the center line
        marker_1 = 0.1 * track_width
        
        # Calculate the direction of the center line based on the closest waypoints
        next_point = waypoints[closest_waypoints[1]]
        prev_point = waypoints[closest_waypoints[0]]
    
        # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
        track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        # Convert to degree
        track_direction = math.degrees(track_direction)
        
        # If current track is straight
        if track_direction == 0:
            # Give higher reward if the car is closer to center line and vice versa
            if distance_from_center <= marker_1:
                if speed < 1.0:
                    reward *= 0.5
                else:
                    reward *= 2.0
            else:
                reward *= 0.5
        else:
            # Calculate the difference between the track direction and the heading direction of the car
            direction_diff = abs(track_direction - heading)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff
        
            # Penalize the reward if the difference is too large
            DIRECTION_THRESHOLD = 10.0
            if direction_diff > DIRECTION_THRESHOLD:
                reward *= 0.5
            else:
                # Cutting corners - Going left
                if track_direction < 0:
                    # Stay left
                    if left:
                        reward *= 2.0
                    else:
                        reward *= 0.5
                else:
                    if left:
                        reward *= 0.5
                    else:
                        reward *= 2.0
    
    return float(reward)