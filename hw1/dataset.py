import os
import cv2

def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    # raise NotImplementedError("To be implemented")
    dataset = list() #宣告一個空的 list 裝 dataset

    for images in os.listdir(dataPath+'/car'):
      # if len(dataset) > 10:
      #   break
      original = cv2.imread(dataPath + '/car/' + images) # 用cv2將原本的圖片讀取進程式
      resized = cv2.resize(original, (36,16))  # resize 圖片到36*16
      gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # 將圖片轉成灰階
      dataset.append((gray , 1)) # 放進 list 當中

    # 再做一次跟上面相同的步驟，但是換成non-car資料夾裡面的圖片
    for images in os.listdir(dataPath+'/non-car'):
      # if len(dataset) > 20:
      #   break
      original = cv2.imread(dataPath + '/non-car/' + images)
      resized = cv2.resize(original, (36,16))
      gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
      dataset.append((gray , 0)) # 因為是 non-car 中的圖片所以 tuple 的第二個 element 要放 0
      
    # End your code (Part 1)
    
    return dataset

