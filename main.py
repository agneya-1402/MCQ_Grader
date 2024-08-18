import cv2
import numpy as np
import utils

webCamFeed = True
pathImage = "MCQ_Checker/test.jpg"
cap = cv2.VideoCapture(1)
cap.set(10,160)
heightImg = 700
widthImg  = 700
questions=5
choices=5
ans= [1,3,5,4,2]
count=0

while True:

    if webCamFeed:success, img = cap.read()
    else:img = cv2.imread(pathImage)

    # RESIZE IMAGE
    img = cv2.resize(img, (widthImg, heightImg)) 
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) 

    # canny edge detect
    imgCanny = cv2.Canny(imgBlur,10,70) 

    try:
        # find contours
        imgContours = img.copy() 
        imgBigContour = img.copy() 
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        # Draw contours
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 
        rectCon = utils.rectContour(contours) 

        # GET CORNER POINTS OF THE BIGGEST RECTANGLE
        biggestPoints= utils.getCornerPoints(rectCon[0]) 
        gradePoints = utils.getCornerPoints(rectCon[1]) 

        

        if biggestPoints.size != 0 and gradePoints.size != 0:
            
            # APPLY WARP PERSPECTIVE
            biggestPoints=utils.reorder(biggestPoints)
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) 
            pts1 = np.float32(biggestPoints) 
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 
            matrix = cv2.getPerspectiveTransform(pts1, pts2) 
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) 

            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20) 
            gradePoints = utils.reorder(gradePoints) 
            ptsG1 = np.float32(gradePoints) 
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) 

            # APPLY THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) 
            imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1]

            boxes = utils.splitBoxes(imgThresh) 
            cv2.imshow("Split Test ", boxes[3])
            countR=0
            countC=0
            myPixelVal = np.zeros((questions,choices)) 
            for image in boxes:
                #cv2.imshow(str(countR)+str(countC),image)
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC]= totalPixels
                countC += 1
                if (countC==choices):countC=0;countR +=1

            # FIND THE USER ANSWERS AND PUT THEM IN A LIST
            myIndex=[]
            for x in range (0,questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])
            print("USER ANSWERS",myIndex)

            # COMPARE VALUES TO FIND CORRECT ANSWERS
            grading=[]
            for x in range(0,questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:grading.append(0)
            print("GRADING",grading)
            score = (sum(grading)/questions)*100 
            print("SCORE",score)

            # DISPLAYING ANSWERS
            utils.showAnswers(imgWarpColored,myIndex,grading,ans)
            utils.drawGrid(imgWarpColored) 
            imgRawDrawings = np.zeros_like(imgWarpColored) 
            utils.showAnswers(imgRawDrawings, myIndex, grading, ans) 
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1) 
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

            # DISPLAY GRADE
            imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) 
            cv2.putText(imgRawGrade,str(int(score))+"%",(70,100)
                        ,cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3) 
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) 
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) 

            # SHOW ANSWERS AND GRADE ON FINAL IMAGE
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)

            # IMAGE ARRAY FOR DISPLAY
            imageArray = ([img,imgGray,imgCanny,imgContours],
                          [imgBigContour,imgThresh,imgWarpColored,imgFinal])
            cv2.imshow("Final Result", imgFinal)
    except:
        imageArray = ([img,imgGray,imgCanny,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

 
    lables = [["Original","Gray","Edges","Contours"],
              ["Biggest Contour","Threshold","Warpped","Final"]]

    stackedImage = utils.stackImages(imageArray,0.5,lables)
    cv2.imshow("Result",stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgFinal)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
