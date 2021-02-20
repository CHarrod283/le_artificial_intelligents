import math
import numpy as npy
import pandas as pd
from scipy import stats

# Ignore deprecation warnings we have no power over
import warnings
warnings.filterwarnings('ignore')

import raceline

def circle_radius(coords):
    # Flatten the list and assign to variables
    x1, y1, x2, y2, x3, y3 = [i for sub in coords for i in sub]

    a = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
    b = (x1**2+y1**2)*(y3-y2) + (x2**2+y2**2)*(y1-y3) + (x3**2+y3**2)*(y2-y1)
    c = (x1**2+y1**2)*(x2-x3) + (x2**2+y2**2)*(x3-x1) + (x3**2+y3**2)*(x1-x2)
    d = (x1**2+y1**2)*(x3*y2-x2*y3) + (x2**2+y2**2) * \
        (x1*y3-x3*y1) + (x3**2+y3**2)*(x2*y1-x1*y2)

    # In case a is zero (so radius is infinity)
    try:
        r = abs((b**2+c**2-4*a*d) / abs(4*a**2)) ** 0.5
    except:
        r = 999

    return r

def track_speeds(path):
    #minimum turn speed found to be 1.6-1.75, wanted to be safe so went with 1.5
    speedCoefficient = 1.5
    lookAhead = 5 # higher look ahead means quicker braking but more computation
    speedArray = []
    for x in range(len(path)):
        speed = 4 # max speed you want the car to go
        for y in range(lookAhead + 1):
            point1 = path[(x + y) % len(path)]
            point2 = path[(x + y + 1) % len(path)]
            point3 = path[(x + y + 2) % len(path)]
            speedMax = speedCoefficient * math.sqrt(circle_radius([point1,point2,point3]))
            if(speed > speedMax):
                speed = speedMax
        speedArray.append(speed)
    return speedArray

def steering_angle(coords):
    radius = circle_radius(coords)
    angle = npy.arcsin(0.165 / radius)
    return angle

def is_left_curve(coords):
    # Flatten the list and assign to variables (makes code easier to read later)
    x1, y1, x2, y2, x3, y3 = [i for sub in coords for i in sub]
    return ((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)) > 0


# Calculate the distance between 2 points
def dist_2_points(x1, x2, y1, y2):
    return abs(abs(x1-x2)**2 + abs(y1-y2)**2)**0.5

def track_radiuses(path):
    left_curves = track_left_curves(path)
    rad = []
    for x in range(len(path)):
        point1 = path[(x) % len(path)]
        point2 = path[(x + 1) % len(path)]
        point3 = path[(x + 2) % len(path)]
        if left_curves[x]:
            rad.append(circle_radius([point1,point2,point3]))
        else:
            rad.append(-1 * circle_radius([point1,point2,point3]))
    return rad

def track_left_curves(path):
    curves = []
    for x in range(len(path)):
        point1 = path[(x) % len(path)]
        point2 = path[(x + 1) % len(path)]
        point3 = path[(x + 2) % len(path)]
        curves.append(is_left_curve([point1,point2,point3]))
    return curves

def track_steering(path):
    dist_wheel = 0.165 
    steering = []
    radiuses = track_radiuses(path)
    for x in range(len(path)):
        steer = npy.arcsin(dist_wheel / radiuses[x])
        steering.append(math.degrees(steer))
    return steering

speeds = track_speeds(raceline.path)
steering = track_steering(raceline.path)

track_data = pd.DataFrame({
    "speed"   : speeds,
    "steering": steering
})

# Find standard deviation so that probability of >15 degrees steering is 5%
steering_sd = -15 / stats.norm.ppf(0.05)

# Create array of noise based on normal distribution
resample_size = 100000
steering_noise = npy.random.normal(0,steering_sd,resample_size)


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans

# all_actions is a DataFrame with 2 columns: speed and steering
X = track_data

# Rescale data with minmax
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

# K-Means Clustering
n_clusters = 19
model = MiniBatchKMeans(n_clusters=n_clusters).fit(X_minmax)

# Interpretable Centroids will represent the action space
X_minmax_fit = minmax_scaler.fit(X)
X_centroids = X_minmax_fit.inverse_transform(model.cluster_centers_)

minSpeed = min(speeds)

X_centroids = npy.concatenate((X_centroids, [[minSpeed, -30.0]]))
X_centroids = npy.concatenate((X_centroids, [[minSpeed, 30.0]]))

centroids_dataframe = pd.DataFrame(X_centroids, columns=['Speeds', 'Steering Angles'])

print("\nAction spaces for track: " + raceline.name)
print(centroids_dataframe)

#creating track data
track_data.insert(0, "coordinate", raceline.path, True)
f = open("dataFile.txt", "w")
f.write("raceline = [\n")
for index, row in track_data.iterrows():
    if(index != len(track_data) - 1):
        f.write("\t[" + str(row['coordinate'][0]) + ", " + str(row['coordinate'][1]) + "],\n")
    else:
        f.write("\t[" + str(row['coordinate'][0]) + ", " + str(row['coordinate'][1]) + "]]\n")

f.write("speeds = [\n")
for index, row in track_data.iterrows():
    if(index != len(track_data) - 1):
        f.write("\t" + str(row['speed']) + ",\n")
    else:
        f.write("\t" + str(row['speed']) + "]\n")

f.write("steering_angles = [\n")
for index, row in track_data.iterrows():
    if(index != len(track_data) - 1):
        f.write("\t" + str(row['steering']) + ",\n")
    else:
        f.write("\t" + str(row['steering']) + "]\n")


