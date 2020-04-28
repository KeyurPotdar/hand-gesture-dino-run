import cv2
import numpy as np
import os

# image dimensions
image_x, image_y = 50, 50


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def create_gesture_data(g_id):
    total_pics = 1000
    cap = cv2.VideoCapture(0)
    # gesture capture box
    x, y, w, h = 300, 50, 350, 350

    create_folder(f"gestures/{g_id}")
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # mask to detect hand
        mask = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 250, 255]))
        res = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=1)

        _, thresh = cv2.threshold(dilation, 30, 255, cv2.THRESH_BINARY)
        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

        if len(contours) > 1:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_image = thresh[y1:y1 + h1, x1:x1 + w1]
                # pad image to make it a square
                if w1 > h1:
                    save_image = cv2.copyMakeBorder(save_image, (w1 - h1) // 2, (w1 - h1) // 2, 0, 0,
                                                    cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_image = cv2.copyMakeBorder(save_image, 0, 0, (h1 - w1) // 2, (h1 - w1) // 2,
                                                    cv2.BORDER_CONSTANT, (0, 0, 0))
                save_image = cv2.resize(save_image, (image_x, image_y))
                cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite(f'gestures/{g_id}/{pic_no}.jpg', save_image)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", frame)
        cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if not flag_start_capturing:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing:
            frames += 1
        if pic_no == total_pics:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    g_id = input("Enter gesture number: ")
    create_gesture_data(g_id=g_id)
