from flask import Flask, request, render_template
import pymysql
import cv2
import os
import numpy as np
import webbrowser

app = Flask(__name__)


@app.route("/")
def start():
    return render_template("login.html")


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            face, rect = detect_face(image)
            if face is not None:
                print(subject_dir_path + "/" + image_name)
                faces.append(face)
                labels.append(label)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
    print(labels)
    return faces, labels


face_recognizer = cv2.face.LBPHFaceRecognizer_create()


def predict(test_img):
    # img=test_img.copy()
    face, rect = detect_face(test_img)
    label, confidence = face_recognizer.predict(face)
    return label


@app.route("/login", methods=["POST", "GET"])
def login():
    username = ""
    password = ""
    if request.method == "POST":
        username = request.form["uname"]
        password = request.form["psw"]

    conn = pymysql.connect('localhost', 'root', '', 'facereg')
    cursor = conn.cursor()
    sql = "select id from login where username='" + username + "' and pass='" + password + "'"
    id = 0
    try:
        cursor.execute(sql)
		# print(cursor.fetchone()[0])
        id = int(cursor.fetchone()[0])
        conn.commit()
    except Exception as e:
        print(e)

    if id == 0:
        return "incorrect username and password"

    faces, labels = prepare_training_data("F:\\opencv\\training_data")
    face_recognizer.train(faces, np.array(labels))
    vid = cv2.VideoCapture(0)
    n = 1
    test_img1 = ""
    while (True):
        ret, frame = vid.read()
        face, rect = detect_face(frame)
        if face is not None:
            test_img1 = frame
            n -= 1

        if cv2.waitKey(1) == ord('Q') or n == 0:
            break

    predicted_label = predict(test_img1)

    if predicted_label == id:


        webbrowser.open("https://drive.google.com/drive/my-drive")

        return "successfully log in"
    else:
        return str(predicted_label)


if __name__ == '__main__':
    app.run(debug=True)