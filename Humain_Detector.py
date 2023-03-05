import math
import shutil
import threading

import cv2
import numpy as np
import os
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from os.path import splitext
from tkinter import filedialog as fd
from tkinter import messagebox
import time

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QToolBar, QAction, QLabel
from PyQt5.QtGui import QPixmap, QImage, QIcon

class Detector:
    ImageExtension=['.jpg','.png','.gif','.bmp','.apng','.avif','.svg','.webp','.ico','.tiff']
    VideoExtension=['.mp4','.avi','.mov','.wmv','.flv','.avchd','.webm','.mkv']
    def __init__(self):
        def askQuestion():
            reply = messagebox.askyesno('confirmation', 'Would you like to use the camera ?')
            if reply == True:
                HumainVideo(0)
            else:
                path=select_file()
                file_name = os.path.basename(path)
                name=os.path.splitext(file_name)[0]
                fileName,fileExtension=splitext(path)
                if fileExtension.lower() in self.ImageExtension:
                    HumainImage(path,name)
                elif fileExtension.lower() in self.VideoExtension:
                    HumainVedioV(path,name)
        def select_file():
            filetypes = (
                ('Image files', '*.jpg;*.png;*.gif;*.bmp;*.apng;*.avif;*.svg;*.webp;*.ico;*.tiff'),
                ('Video files', '*.mp4;*.avi;*.mov;*.wmv;*.flv;*.avchd;*.webm;*.mkv')
            )
            filename = fd.askopenfilename(
                title='Open a file',
                initialdir='/',
                filetypes=filetypes)
            return filename

        def runInThread(target):
            thread=threading.Thread(target=target)
            thread.start()

        root = Tk()
        canvas = Canvas(root, width=800, height=400)
        canvas.pack(expand=True, fill="both")
        btn = tk.Button(root, command=lambda: runInThread(askQuestion),
                        activebackground="#33B5E5")
        btn.pack(expand=True)
        btn.place(height=100, width=100, x=20, y=20)
        img1 = ImageTk.PhotoImage(Image.open("images\\camera.png"))
        btn.config(image=img1, width=200, height=100)
        img2 = ImageTk.PhotoImage(Image.open("images\\bk.jpg"))
        canvas.create_image(2, 2, anchor=NW, image=img2)
        canvas.create_window(400, 376, height=86, width=133, anchor='s', window=btn)
        root.mainloop()

class HumainImage:
    def __init__(self, path,filename):
        img = cv2.imread(path)
        if os.path.exists("peoples\\"+filename):
            os.chmod("peoples\\"+filename, 0o777)
            shutil.rmtree("peoples\\"+filename)
        os.mkdir("peoples\\"+filename)

        if os.path.exists("peoples\\"+filename+"_distance.txt"):
            os.chmod("peoples\\"+filename+"_distance.txt", 0o777)
            os.remove("peoples\\"+filename+"_distance.txt")
        dist=open("peoples\\"+filename+"_distance.txt","w")

        classNames = []
        classFile = 'coco.names'

        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightpath = 'frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightpath, configPath)
        net.setInputSize(320, 230)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        print(classIds, bbox)
        count=0
        listPeople=[]
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classNames[classId - 1] == "person":
                    (x,y,w,h)=box
                    listPeople.append(People('People ' + str(count), (x + x + w)/2,(y + y + h)/2))
                    cv2.imwrite('peoples\\'+filename+'\\people'+str(count+1)+'.png', img[box[1]-25:box[1]+box[3], box[0]:box[0]+box[2]])
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
                    cv2.putText(img, classNames[classId - 1]+str(count), (box[0] + 10, box[1] + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=1)
                    count=count+1

            if count > 0:
                n = len(listPeople)
                for j in range(n):
                    for i in range(j,n):
                        people = listPeople[i]
                        people2 = listPeople[j]
                        distancePiX = math.sqrt((people.abs - people2.abs) ** 2 + (people.ord - people2.ord) ** 2)
                        distanceCm = distancePiX * 2.54 / 240
                        if people2.name != people.name:
                            dist.write("The distance between " +people2.name+ " and " + people.name + " is " + str(distanceCm) + " m\n")
        dist.close()
        print("number of person is "+str(count))
        cv2.imshow('Output', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class HumainVideo:
    def __init__(self, path):
        self.mask = cv2.imread("images\\mask1.png")

        # Create the GUI application
        app = QApplication(sys.argv)
        # Create the main window
        window = QMainWindow()
        # Create the toolbar
        toolbar = QToolBar()
        # Create a action for each icon
        icon1_action = QAction("icon 1")
        icon1_action.setIcon(QIcon("images\\mask1.png"))
        icon1_action.triggered.connect(lambda: HumainVideo.set_mask(self,"images\\mask1.png",cam, net, classNames, face_cascade, out, label, app, window))
        icon2_action = QAction("Icon 2")
        icon2_action.setIcon(QIcon("images\\mask2.png"))
        icon2_action.triggered.connect(lambda: HumainVideo.set_mask(self,"images\\mask2.png",cam, net, classNames, face_cascade, out, label, app, window))
        icon3_action = QAction("Icon 3")
        icon3_action.setIcon(QIcon("images\\mask3.png"))
        icon3_action.triggered.connect(lambda: HumainVideo.set_mask(self,"images\\mask3.png",cam, net, classNames, face_cascade, out, label, app, window))
        icon4_action = QAction("Icon 4")
        icon4_action.setIcon(QIcon("images\\mask4.png"))
        icon4_action.triggered.connect(lambda: HumainVideo.set_mask(self,"images\\mask4.png",cam, net, classNames, face_cascade, out, label, app, window))
        icon5_action = QAction("icon 5")
        icon5_action.setIcon(QIcon("images\\mask5.png"))
        icon5_action.triggered.connect(
            lambda: HumainVideo.set_mask(self, "images\\mask5.png", cam, net, classNames, face_cascade, out, label, app,
                                         window))
        icon6_action = QAction("icon 6")
        icon6_action.setIcon(QIcon("images\\mask6.png"))
        icon6_action.triggered.connect(
            lambda: HumainVideo.set_mask(self, "images\\mask6.png", cam, net, classNames, face_cascade, out, label, app,
                                         window))
        icon7_action = QAction("icon 7")
        icon7_action.setIcon(QIcon("images\\mask7.png"))
        icon7_action.triggered.connect(
            lambda: HumainVideo.set_mask(self, "images\\mask7.png", cam, net, classNames, face_cascade, out, label, app,
                                         window))
        icon8_action = QAction("icon 8")
        icon8_action.setIcon(QIcon("images\\mask8.png"))
        icon8_action.triggered.connect(
            lambda: HumainVideo.set_mask(self, "images\\mask8.png", cam, net, classNames, face_cascade, out, label, app,
                                         window))
        icon9_action = QAction("icon 9")
        icon9_action.setIcon(QIcon("images\\mask9.png"))
        icon9_action.triggered.connect(
            lambda: HumainVideo.set_mask(self, "images\\mask9.png", cam, net, classNames, face_cascade, out, label, app,
                                         window))
        icon10_action = QAction("icon 10")
        icon10_action.setIcon(QIcon("images\\mask10.png"))
        icon10_action.triggered.connect(
            lambda: HumainVideo.set_mask(self, "images\\mask10.png", cam, net, classNames, face_cascade, out, label, app,
                                         window))

        # Add the actions to the toolbar
        toolbar.addAction(icon1_action)
        toolbar.addAction(icon2_action)
        toolbar.addAction(icon3_action)
        toolbar.addAction(icon4_action)
        toolbar.addAction(icon5_action)
        toolbar.addAction(icon6_action)
        toolbar.addAction(icon7_action)
        toolbar.addAction(icon8_action)
        toolbar.addAction(icon9_action)
        toolbar.addAction(icon10_action)

        # Add the toolbar to the main window
        window.addToolBar(toolbar)
        # Create a label to show the video
        label = QLabel()
        # Set the label as the central widget
        window.setCentralWidget(label)
        # Show the main window
        window.show()

        cam = cv2.VideoCapture(path)
        cam.set(3, 740)
        cam.set(4, 580)
        # the output will be written to output.avi
        if os.path.exists("peoples\\camera"):
            os.chmod("peoples\\camera", 0o777)
            shutil.rmtree("peoples\\camera")
        os.mkdir("peoples\\camera")
        out = cv2.VideoWriter(
            'peoples\\camera\\output.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            15.,
            (640, 480))

        classNames = []
        classFile = 'coco.names'

        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightpath = 'frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightpath, configPath)
        net.setInputSize(320, 230)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        # Load the cascades for face detection
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        HumainVideo.set_mask(self,"images\\mask1.png",cam, net, classNames, face_cascade, out, label, app, window)

        cam.release()
        # and release the output
        out.release()
        cv2.destroyAllWindows()
        # Exit the GUI application
        sys.exit()

    def set_mask(self,pathIcon,cam,net,classNames,face_cascade,out,label,app,window):
        self.mask = cv2.imread(pathIcon)
        while True:
            success, img = cam.read()
            classIds, confs, bbox = net.detect(img, confThreshold=0.5)
            print(classIds, bbox)
            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    if classNames[classId - 1] == "person":
                        # Detect faces in the image
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
                        # For each face, add the mask as a overlay
                        for (x, y, w, h) in faces:
                            self.mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_LINEAR)
                            dist = cv2.addWeighted(img[y:y+h,x:x+w], 0.5, self.mask, 0.5, 0)
                            img[y:y + h,x:x+w]=dist
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
                        cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=1)

            # Write the output video
            out.write(img.astype('uint8'))
            # Convert the frame to a QImage
            image = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
            # Set the image as the pixmap of the label
            label.setPixmap(QPixmap.fromImage(image))
            # Update the label
            label.update()
            # Process the events of the GUI
            app.processEvents()

            # Check if the user closes the window
            if window.isHidden():
                break
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

class People:
    def __init__(self,Name,x,y):
        self.name=Name
        self.abs=x
        self.ord=y

class HumainVedioV:
    def __init__(self,video_src,filename):
        os.chdir(r"C:\Users\Admin\PycharmProjects\pythonProject2")

        # line a
        ax1 = 350
        ay = 400
        ax2 = 900

        # video ....
        cap = cv2.VideoCapture(video_src)
        if os.path.exists("venv\\peoples\\"+filename):
            os.chmod("venv\\peoples\\"+filename, 0o777)
            shutil.rmtree("venv\\peoples\\"+filename)
        os.mkdir("venv\\peoples\\"+filename)

        if os.path.exists("venv\\peoples\\"+filename+"_distance.txt"):
            os.chmod("venv\\peoples\\"+filename+"_distance.txt", 0o777)
            os.remove("venv\\peoples\\"+filename+"_distance.txt")
        dist=open("venv\\peoples\\"+filename+"_distance.txt","w")

        if os.path.exists("venv\\peoples\\"+filename+"_speed.txt"):
            os.chmod("venv\\peoples\\"+filename+"_speed.txt", 0o777)
            os.remove("venv\\peoples\\"+filename+"_speed.txt")
        speedF=open("venv\\peoples\\"+filename+"_speed.txt","w")

        classFile = 'venv\\coco.names'
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'venv\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightpath = 'venv\\frozen_inference_graph.pb'

        car_cascade = cv2.dnn_DetectionModel(weightpath, configPath)
        car_cascade.setInputSize(320, 230)
        car_cascade.setInputScale(1.0 / 127.5)
        car_cascade.setInputMean((127.5, 127.5, 127.5))
        car_cascade.setInputSwapRB(True)

        count = 0
        start_video=time.time()
        end_frame=0
        listPeople = []
        while True:
            start_frame=time.time()
            ret, img = cap.read()
            if (type(img) == type(None)):
                break

            classIds, confs, bbox = car_cascade.detect(img, confThreshold=0.5)

            if len(classIds) != 0:
                for classId, confidence, (x, y, w, h) in zip(classIds.flatten(), confs.flatten(), bbox):
                    if classNames[classId - 1] == "person":
                        while int(ay) == int((y + y + h) / 2):
                            listPeople.append(People('People ' + str(count), (x + x + w)/2,(y + y + h)/2))
                            if count>0:
                                n = len(listPeople)
                                for i in range(n-1):
                                    people = listPeople[i]
                                    distancePiX=math.sqrt((people.abs-x)**2+(people.ord-y)**2)
                                    distanceCm=distancePiX*2.54/240
                                    if ('People '+str(count))!=people.name:
                                        dist.write("The distance between people "+str(count)+" and "+people.name+" is "+str(distanceCm)+" m\n")
                            time_passe=(time.time()-start_video)-end_frame
                            speed = (30*3.6)/time_passe
                            speedF.write("person Number " + str(count) + " Speed: " + str(speed) + " KM/H \n")
                            cv2.imwrite('venv\\peoples\\'+filename+'\\people' + str(count)+'_'+str(speed) + '.png',
                                        img[y-25:y + h, x:x + w])
                            count = count + 1
                            break
                        cv2.rectangle(img, (x,y,w,h), color=(0, 255, 0), thickness=1)
                        cv2.putText(img, classNames[classId - 1]+str(count), (x + 10, y + 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=1)

            cv2.imshow('video', img)
            end_frame=end_frame+time.time()-start_frame
            if cv2.waitKey(33) == 27:
                break
        dist.close()
        cap.release()  # close cam
        cv2.destroyAllWindows()  # close window
det=Detector()