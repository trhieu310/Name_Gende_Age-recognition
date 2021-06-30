import tkinter as tk
from tkinter import *
from tkinter import ttk, RIGHT, messagebox
from PIL import ImageTk, Image
from imutils import paths 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils.video import VideoStream
import imutils
import pickle
import numpy as np
import cv2
import os
import time
import threading

form = tk.Tk()
form.title("Face Recognition")
form.geometry("700x650+300+20")

name_path = StringVar()

ageList = ['(0-2)', '(4-6)', '(8 -12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']

cap = cv2.VideoCapture(0)

def create_path():
    global newpath, name_input
    
    name_input = name_path.get()

    if(name_input == ""):
        infox="Try again"
        messagebox.showinfo(title = "Information", message = infox)
    else:
        if not os.path.exists('dataset/'):
            os.makedirs('dataset/')
            print("[INFO] Create dataset completed...")

        newpath = r'dataset/' + name_input
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            print("[INFO] Create "+name_input+" completed...")

def stop():
    cap.release()

def take_pic():
    count = 0
    while (True):
        ret, frame = cap.read()
        img = frame.copy()
        count += 1
        cv2.imwrite(newpath+"/" + str(count) + ".jpg", img)
        print(count)

        time.sleep(0.3)
        if count >= 30:
            print("[INFO] Done! You can close the program...")
            stop()
            return False

def show_frame():
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    prevImg = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=prevImg)
    view_train.imgtk = imgtk
    view_train.configure(image=imgtk)
    view_train.after(10, show_frame)

def capture():
    t1 = threading.Thread(target=take_pic, args=())
    t2 = threading.Thread(target=show_frame, args=())
    t1.start()
    t2.start()

def extract_faces():
    print("[INFO] loading face detector...")
    protoPath = "deploy.prototxt"
    modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    print("[INFO] loading face recognizer...")
    embedding_model = "openface_nn4.small2.v1.t7"
    embedder = cv2.dnn.readNetFromTorch(embedding_model)

    print("[INFO] quantifying faces...")
    dataset = r'dataset'
    imagePaths = list(paths.list_images(dataset))

    knownEmbeddings = []
    knownNames = []

    total = 0
    confiden = 0.5

    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=900)
        (h, w) = image.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > confiden:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    embeddings = "embeddings.pickle"
    f = open(embeddings, "wb")
    f.write(pickle.dumps(data))
    f.close()

def train():
    print("[INFO] loading face embeddings...")
    embeddings = "embeddings.pickle"
    data = pickle.loads(open(embeddings, "rb").read())

    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    recognize = "recognize.pickle"
    f = open(recognize, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    
    les = "le.pickle"
    f = open(les, "wb")
    f.write(pickle.dumps(le))
    f.close()

    print("[INFO] training finished...")

def trainer():
    extract_faces()
    train()
def recognition():
    print("[INFO] loading face detector...")
    protoPath = "deploy.prototxt"
    modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    print("[INFO] loading face recognizer...")
    embedding_model = "openface_nn4.small2.v1.t7"
    embedder = cv2.dnn.readNetFromTorch(embedding_model)

    print("[INFO] loading age detector model...")
    aprototxtPath = "age_deploy.prototxt"
    aweightsPath = "age_net.caffemodel"
    ageNet = cv2.dnn.readNet(aprototxtPath, aweightsPath)

    print("[INFO] loading gender detector model...")
    gprototxtPath = "gender_deploy.prototxt"
    gweightsPath = "gender_net.caffemodel"
    genderNet = cv2.dnn.readNet(gprototxtPath, gweightsPath)

    recognize = "recognize.pickle"
    les = "le.pickle"

    recognizer = pickle.loads(open(recognize, "rb").read())
    le = pickle.loads(open(les, "rb").read())

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    results = []

    confiden = 0.5

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]
            if confidence > confiden:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                name = le.classes_[j]
                
                ageNet.setInput(blob)
                apreds = ageNet.forward()
                a = apreds[0].argmax()
                age = ageList[a]

                genderNet.setInput(blob)
                gpreds = genderNet.forward()
                g = gpreds[0].argmax()
                gender = genderList[g]
            
                text = "{}, {}, {}".format(name, gender, age)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            else:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue
                
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                text = "Unknow"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)              

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            vs.stop()
            cv2.destroyAllWindows()
            break

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)

tab_parent.add(tab1, text="Main")
tab_parent.add(tab2, text="Train")

photo = ImageTk.PhotoImage(Image.new("RGB", (750, 500), "white"))
#== Khai báo phần tử cho tab Main
lb_Main = tk.Label(tab1, text="Face Recognition", font=("Helvetica", 25))
view = tk.Label(tab1, image= photo)

fr_but = Frame(tab1, borderwidth=1)
start_button = tk.Button(fr_but, text="Start", command=recognition)
stop_button = tk.Button(fr_but, text="Stop", command = stop) 

#== Đặt vị trí cho phần tử tab Main
lb_Main.grid(row = 0, column = 2, dax = 5, pady = 5)
view.grid(row = 1, column = 2, padx = 5, pady = 5)

fr_but.grid(row = 2, column = 2, padx = 5, pady = 5)
stop_button.pack(side=RIGHT, ipadx = 15, padx=5, pady=5)
start_button.pack(side=RIGHT, ipadx = 15, padx=5, pady=5)

#== Khai báo phần tử cho tab Train
lb_Name = tk.Label(tab2, text="Name:")
entry_Name = tk.Entry(tab2, textvariable = name_path)
get_value = tk.Button(tab2, text="Get Name", command=create_path)

view_train = tk.Label(tab2, image=photo)

fr__but = Frame(tab2, borderwidth=1)
sample_button = tk.Button(fr__but, text="Get Sample", command=capture)
train_button = tk.Button(fr__but, text="Train", command=trainer) 
#== Đặt vị trí cho phần tử tab Detect

lb_Name.grid(row = 0, column = 6,padx = 5, pady = 16, sticky = E)
entry_Name.grid(row = 0, column = 7, padx = 5, pady = 5, sticky = EW)
get_value.grid(row = 0, column = 8, sticky = W)

view_train.grid(row = 1, column = 1, columnspan=15, padx = 5, pady = 5)

fr__but.grid(row = 2, column = 7, pady = 5)
train_button.pack(side=RIGHT,ipadx = 15, padx=5, pady=5)
sample_button.pack(side=RIGHT,ipadx = 5, padx=5, pady=5)

tab_parent.pack(expand=1, fill='both')

form.mainloop()