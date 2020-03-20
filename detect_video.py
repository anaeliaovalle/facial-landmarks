import imutils
from imutils import face_utils
import dlib
import cv2
import argparse

def capture_face(args): 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_model)
    
    #  start video capture 
    cap = cv2.VideoCapture(0)
    
    while True:
        #  load in image taken from video capture
        _, image = cap.read()

        #  set image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #  detect faces in the grayscale image
        rects = detector(gray, 0)

        #  loop over each face detected
        for rect in rects:
            
            #  get the 68 predicted landmarks for the face
            shape = predictor(gray, rect)

            #  convert the (x,y) coordinate tuples into a numpy array
            shape = imutils.face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            (x, y, w, h) = imutils.face_utils.rect_to_bb(rect)

            #  draw a rectange in the face make it red
            cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2) 

            #  draw each facial landmarks (x,y) point onto the face
            for (x, y) in shape:
                cv2.circle(img=image, center=(x, y), radius=1, color=(0, 255, 0), thickness=-1)


		#  Show the image
        cv2.imshow("Output", image)
        
        # hold frame for 2 milliseconds and make it num-loc insensitive
        wait_check = cv2.waitKey(2) & 0xFF

        # press q to quit
        if wait_check == ord('q'):
            break
        
    cv2.destroyAllWindows()  # destroy gui window and make new one upon new pic
    cap.release()  # closes video


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape-model', help=".dat file that models shape", default='shape_predictor_68_face_landmarks.dat') 
    args = parser.parse_args()
    capture_face(args)