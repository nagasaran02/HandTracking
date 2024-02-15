import cv2
import mediapipe as mp
import time



cap = cv2.VideoCapture(0)
mphand = mp.solutions.hands#initialize the hand module
hands = mphand.Hands()#parameters are static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
mpDraw = mp.solutions.drawing_utils#initialize the drawing module
pTime = 0#previous time
cTime = 0#current time


while True:
    success, img = cap.read()#read the image from the camera and store it in img variable.success is a boolean variable which is true if the image is read successfully and false if not.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# convert the image to RGB
    results = hands.process(imgRGB)#process the image and store the results in the results variable
    #print(results)
    #print(results.multi_hand_landmarks)#print the landmarks of the hand
    if results.multi_hand_landmarks:#if there are multiple hands
        for handLms in results.multi_hand_landmarks:#for each hand
            for id, lm in enumerate(handLms.landmark):#for each landmark in the hand.lm is the landmark and id is the id of the landmark
                #print(id, lm)#print the id and the landmark .Here the image is in the form of pixels and the landmarks are in the form of ratios
                h, w, c = img.shape#height, width, channel
                cx, cy = int(lm.x*w), int(lm.y*h)#center x, center y. convert the ratios to pixels by multiplying with the width and height
                print(id, cx, cy)#print the id and the center x and y
                if id ==14:#if the id is 0 we can try with different id's we get differtent landmarks
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)#draw a circle at the center x and y with radius 15 and color (255, 0, 255)
            mpDraw.draw_landmarks(img, handLms, mphand.HAND_CONNECTIONS)#draw the landmarks and the connections between them on the image.here handLms is the landmarks of the hand and HAND_CONNECTIONS are the connections between the landmarks

    cTime = time.time()#get the current time
    fps = 1/(cTime-pTime)#calculate the frames per second
    pTime = cTime#set the previous time to the current time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)#put the fps on the image by using the putText function.(10,70 ) is the position of the text, FONT_HERSHEY_PLAIN is the font of the text, 3 is the size of the text, (255, 0, 255) is the color of the text and 3 is the thickness of the text.
    cv2.imshow("Image", img)
    cv2.waitKey(1)