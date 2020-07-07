# Social_Distance_Detector-COVID-19-
# link for weights of yolo download- https://pjreddie.com/darknet/yolo/
COVID-19 social distancing detector using OpenCV, Deep Learning, and Computer Vision.

 OpenCV and deep learning can be used to implement a social distancing detector.

We’ll then review our project directory structure including:

    Our configuration file used to keep our implementation neat and tidy
    Our detect_people
    utility function, which detects people in video streams using the YOLO object detector
    Our Python driver script, which glues all the pieces together into a full-fledged OpenCV social distancing detector

We’ll wrap up the post by reviewing the results, including a brief discussion on limitations and future improvements.
What is social distancing?
Figure 1: Social distancing is important in times of epidemics and pandemics to prevent the spread of disease. Can we build a social distancing detector with OpenCV? (image source)

Social distancing is a method used to control the spread of contagious diseases.

As the name suggests, social distancing implies that people should physically distance themselves from one another, reducing close contact, and thereby reducing the spread of a contagious disease (such as coronavirus):
Figure 2: Social distancing is crucial to preventing the spread of disease. Using computer vision technology based on OpenCV and YOLO-based deep learning, we are able to estimate the social distance of people in video streams. (image source)

Social distancing is not a new concept, dating back to the fifth century (source), and has even been referenced in religious texts such as the Bible:

    And the leper in whom the plague is … he shall dwell alone; [outside] the camp shall his habitation be. — Leviticus 13:46

Social distancing is arguably the most effective nonpharmaceutical way to prevent the spread of a disease — by definition, if people are not close together, they cannot spread germs.
Using OpenCV, computer vision, and deep learning for social distancing
Figure 3: The steps involved in an OpenCV-based social distancing application.

We can use OpenCV, computer vision, and deep learning to implement social distancing detectors.

The steps to build a social distancing detector include:

    Apply object detection to detect all people (and only people) in a video stream (see this tutorial on building an OpenCV people counter)
    Compute the pairwise distances between all detected people
    Based on these distances, check to see if any two people are less than N pixels apart

For the most accurate results, you should calibrate your camera through intrinsic/extrinsic parameters so that you can map pixels to measurable units.

An easier alternative (but less accurate) method would be to apply triangle similarity calibration (as discussed in this tutorial).

Both of these methods can be used to map pixels to measurable units.

Finally, if you do not want/cannot apply camera calibration, you can still utilize a social distancing detector, but you’ll have to rely strictly on the pixel distances, which won’t necessarily be as accurate.


