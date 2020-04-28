import cv2
import numpy as np
from keras.models import load_model
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

model = load_model('hand_gesture.h5')


def main():
    cap = cv2.VideoCapture(0)
    # load dino run game
    driver = webdriver.Chrome('./chromedriver.exe')
    driver.get('chrome://dino/')
    body = driver.find_element_by_tag_name("body")
    x, y, w, h = 300, 50, 350, 350

    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 250, 255]))
        res = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=1)
        _, thresh = cv2.threshold(dilation, 30, 255, cv2.THRESH_BINARY)

        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w1, h1 = cv2.boundingRect(contour)
                image = thresh[y:y + h1, x:x + w1]
                if w1 > h1:
                    image = cv2.copyMakeBorder(image, (w1 - h1) // 2, (w1 - h1) // 2, 0, 0,
                                               cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    image = cv2.copyMakeBorder(image, 0, 0, (h1 - w1) // 2, (h1 - w1) // 2,
                                               cv2.BORDER_CONSTANT, (0, 0, 0))
                image = cv2.resize(image, (50, 50))
                pred = keras_predict(model, image)
                print(pred)
                # jump if predicted class is 0 (open hand)
                if pred == 0:
                    body.send_keys(Keys.SPACE)

        x, y, w, h = 300, 50, 350, 350
        cv2.imshow('Frame', frame)
        cv2.imshow('Contours', thresh)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    driver.close()


def keras_predict(model, image):
    image_x, image_y = 50, 50
    image = np.reshape(image, (1, image_x, image_y, 1))
    return model.predict(image)[0][0]


keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
main()
