import cv2
import face_recognition
import pickle
import os

#importing the mode images into a list
folderModePath = "C:/Users/User/PycharmProjects/pythonProject1/images"
pathList = os.listdir(folderModePath)
print(pathList)
imgList = []
studentIds = []
for path in pathList :
    imgList.append(cv2.imread(os.path.join(folderModePath,path)))
    print(os.path.splitext(path)[0])
    studentIds.append(os.path.splitext(path)[0])

#print(len(imgList))
def findEncodings(imagesList) :
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode  = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
#print(encodeListKnown)
encodeListKnown_Ids = [encodeListKnown,studentIds]
print("Encoding Complete.")
file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnown_Ids,file )
file.close()
print("File saved.")