import cv2

def main():
    name = input("Input name:")
    cap = cv2.VideoCapture(0)    #打开摄像头
    while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)   #生成摄像头窗口
    key = cv2.waitKey(3)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #如果按下q 就截图保存并退出
        cv2.imwrite("test_img/{}.jpg".format(name), frame)
        break
    elif key == 27:
        break
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
