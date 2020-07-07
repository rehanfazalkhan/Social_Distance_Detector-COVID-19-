import cv2
import numpy as np
# min. probability to filter weak detections
min_cof = 0.3
# min. threshold when applying non-maximum suppression
min_thr = 0.3

def detect_people_classes(frame, net, ln, personIdx=0):
    # dimensions of frames
    (h, w) = frame.shape[:2]

    # results consist of
    # (1) the person prediction probability
    # (2) bounding box coordinates for the detection, and
    # (3) the centroid of the object.
    results = []

    # construct the blob from input frame & perform forward pass of YOLO object detector return bounding box & proba
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    # initialize the lists of bounding box, centroid & confidence
    boxes = []
    centroids = []
    confidences = []

    # loop over each layer of outputs
    for output in layer_outputs:
        # loop over each detections
        for detection in output:
            # class ID & Probability
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter the weak detection & also the detection is " person class"
            if classID == personIdx and confidence > min_cof:
                # scale the bounding box relative to size of the frame
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                # use the centre(x,y) coordinate to drive width & height
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((center_x, center_y))
                confidences.append(float(confidence))

    # apply non-max-suppression to suppress weak , overlapping bounding box
    idx = cv2.dnn.NMSBoxes(boxes, confidences, min_cof, min_thr)

    # ensure at lest one detection exist
    if len(idx) > 0:
        # loop over index we are keeping
        for i in idx.flatten():
            # extract the bounding box
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update results consists of the person probability, centroids & bounding box
            r = (confidences[i], (x, y, x+w, y+h), centroids[i])
            results.append(r)

    return results














