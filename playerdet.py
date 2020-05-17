import cv2
import numpy as np

path_to_video=" "  

#Tuning parameters:
minContourArea=100
minSolidity=0.55
minAspectRatio=1.6
maxAspectRatio=5
    	

video=cv2.VideoCapture(path_to_video) #load video
startFrame=0 # start from some frame
video.set(cv2.CAP_PROP_POS_FRAMES, startFrame) 

while True:

	#capturing frames and resizing
	ret,frame=video.read()
	frame=cv2.resize(frame,(640,480))
	

	#convert to HSV colorspace
	hsv_img=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 

	# green color range for football field in HSV
	lower_green = np.array([25, 48,68],dtype='uint8')
	upper_green = np.array([50,255,195],dtype='uint8')

	#masking the football field black
	mask=cv2.inRange(hsv_img,lower_green,upper_green) 
	mask=np.invert(mask) 	
	
	# performing morphological opening
	kernel_open=np.ones((4,4),np.uint8)
	mask=cv2.morphologyEx(mask.copy(),cv2.MORPH_OPEN,kernel_open)
	
	#getting contours from mask
	contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]  
	contours=np.array(contours)

	#filter by contour area
	areaFilt=np.array([True if cv2.contourArea(cnt)>minContourArea else False for cnt in contours],dtype=bool)
	contours=contours[areaFilt]
	
	#filter by solidity for removing white field lines
	temp=[]
	for cnt in contours:
		area = cv2.contourArea(cnt)
		hull = cv2.convexHull(cnt)
		hull_area = cv2.contourArea(hull)
		solidity = float(area)/hull_area
		if solidity>minSolidity:
			temp.append(cnt)

	contours=temp

	boundingRects=[cv2.boundingRect(cnt) for cnt in contours ]

	#filter by aspect ratio
	boundingRects=[[x,y,w,h] for x,y,w,h in boundingRects if h/w>minAspectRatio and h/w<maxAspectRatio ]
	
	
	
	#list holding crops of players and coordinates
	players=[]

	#drawing boxes around detected players and adding crops and coordinates to players list
	for (x,y,w,h) in boundingRects:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		players.append(frame[y-25:y+h+25,x-25:x+w+25])
	
	
	#display frames with detections
	cv2.imshow('detections',frame)
	key=cv2.waitKey(10)
	if key==ord('q'):
		break
	
	
		
#cleanup
cv2.destroyAllWindows()
video.release()	
	
		
		


	



