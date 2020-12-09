import numpy as np

def cal_distance(a, b):
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])


def update_marker():
    # Iterate all data
    for i in range(10):
        print("data is: ", data[i])
        newMarker = -1
        minDistance = cal_distance(data[i], centers[0])
        # Find the cluster that has minimal distance
        for j in range(k):
            dis = cal_distance(data[i], centers[j])
            print("center is: ", centers[j], "distance is: ", dis)
            if dis <= minDistance:
                newMarker = j
        # update the marker of this point
        markers[i] = newMarker
        print("marker is: ", newMarker)


# Find the points in cluster 1
# update the center of cluster 1
def update_center():
    for i in range(3):
        size = 0
        X = []
        Y = []
        for j in range(10):
            if markers[j] == i:
                X.append(data[j][0])
                Y.append(data[j][1])
                size += 1
        print(X)
        print(Y)
        centers[i][0] = np.mean(np.array(X))
        centers[i][1] = np.mean(np.array(Y))
        print(centers[i])






k = 3

data = np.array([[5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [5.0, 3.0],
                 [4.9, 3.1], [6.7, 3.1], [5.1, 3.8], [6.0, 3.0]])

centers = np.array([[6.2,3.2], [6.6,3.7], [6.5,3.0]])

markers = np.zeros(10)



for i in range(4):
    update_marker()
    update_center()
    print("iteration : ", i+1)
    print("center of RED is", centers[0, :])
    print("center of GREEN is", centers[1, :])
    print("center of BLUE is", centers[2, :])
    print("--------------------------------------------------")