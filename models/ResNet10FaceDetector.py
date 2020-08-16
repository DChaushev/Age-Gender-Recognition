import cv2
import numpy as np


class ResNet10FaceDetector:
    network = "models/network.prototxt"
    weights = "models/res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(self.network, self.weights)

    def detect_faces(self, image):
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (103.93, 116.77, 123.68))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        (h, w) = image.shape[:2]

        faces = []

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            # ( (x1, y1, x2, y2), confidence)
            faces.append((box.astype("int"), detections[0, 0, i, 2]))

        return faces
