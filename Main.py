import cv2
import time
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from models.ResNet10FaceDetector import ResNet10FaceDetector
from models.AgePredictor import AgePredictor
from models.GenderPredictor import GenderPredictor
from imutils.video import VideoStream


FRAME_NAME = "Age and gender prediction"


def main():
    vs = VideoStream(src=0).start()

    face_detector = ResNet10FaceDetector()
    age_predictor = AgePredictor()
    gender_predictor = GenderPredictor()

    while True:
        frame = vs.read()

        faces = face_detector.detect_faces(frame)

        for face in faces:
            mark_face_on_frame(frame, face, age_predictor, gender_predictor)

        cv2.imshow(FRAME_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        # ESC key
        if key == 27:
            break
        # Window closed
        if cv2.getWindowProperty(FRAME_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    vs.stop()


def mark_face_on_frame(frame, face, age_predictor, gender_predictor):
    start_x, start_y, end_x, end_y, confidence = get_coordinates(face)

    face_image = frame[start_y:end_y, start_x:end_x]
    image = prepare_image(face_image, (80, 60), True)

    age = predict(age_predictor, image, 'age')
    gender = predict(gender_predictor, image, 'gender')

    draw_bounding_box(age, gender, frame, start_x, start_y, end_x, end_y)


def prepare_image(face_image, size, grayscale=False):
    image = cv2.resize(face_image, size)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def get_coordinates(face):
    confidence = face[1]
    (start_x, start_y, end_x, end_y) = face[0]

    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0

    print("Found face ", face[0], " with confidence ", confidence)

    return start_x, start_y, end_x, end_y, confidence


def predict(predictor, image, target_class):
    start = time.time()
    result = predictor.predict(image)
    end = time.time()

    print("Time to predict ", target_class, ": ", end - start)
    print(target_class, ': ', result)

    return result


def draw_bounding_box(age, gender, frame, start_x, start_y, end_x, end_y):
    # draw the bounding box with the corresponding text
    text = "Age: {}".format(int(age))

    #                      B   G   R                          B   G   R
    bounding_box_color = (255, 213, 0) if gender == 1 else (244, 132, 255)

    y = start_y - 10 if start_y - 10 > 10 else start_y + 10
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),
                  bounding_box_color, 2)
    cv2.putText(frame, text, (start_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bounding_box_color, 2)


if __name__ == "__main__":
    main()
