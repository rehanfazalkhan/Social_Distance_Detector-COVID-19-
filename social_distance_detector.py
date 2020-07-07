import cv2
import imutils
import numpy as np

from detect_people import detect_people_classes
from scipy.spatial import distance as dist

# load the COCO class labels YOLO model was trained on
labels = open("coco.names").read().strip().split("\n")

# load model weights and model configuration
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# GPU use
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# get output layer names
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

# video input to detect
video = cv2.VideoCapture("3.mp4")

# loop over the frame
while True:
    true, frame= video.read()

    # if frame where not grabbed break the loop
    if not true:
        break

    # resize the frame & detect only people class
    frame = imutils.resize(frame, width= 700)
    results = detect_people_classes(frame, net, ln, personIdx=labels.index("person"))

    # initialize the set of index that violate the minimum social distance
    violate = set()

    # ensure that at least  2 people detections ( require pair wise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the Euclidean distance between all pairs of centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over upper triangular of distance matrix (matrix is symmetrical)
        for i in range(0, D.shape[0]):
            for j in  range(i+1, D.shape[1]):
                # check if distance between any two centroids pairs is less than the configure no of pixels
                if D[i, j]  < 50:
                    # update our violation set the index of centroids pairs
                    violate.add(i)
                    violate.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bbox and centroids coordinates
        (start_x, start_y, end_x, end_y) = bbox
        (cX, cY) = centroid
        # for no violation
        color = (0, 255, 0)

        # if index pair exist within the violation set ,then update the color
        if  i in violate:
            color = (0, 0, 255)

        # draw bounding box around the person
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # draw total no of social distance violation in output frame
    text = "Social Distance Violation {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,0,0), 3)

    # show the output frame
    cv2.imshow("frame", frame)
    key =cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
















