import os

temproot = "C:/Users/cclinlab/Desktop/Temp/img"
temptarget = "C:/Users/cclinlab/Desktop/Temp/mask"
root = "C:/Users/cclinlab/Desktop/1119/mask_ro"
if __name__=="__main__":
    namelist = os.listdir("C:/Users/cclinlab/Desktop/Temp/img")
    masklist = os.listdir("C:/Users/cclinlab/Desktop/1119/mask_ro")
    for file in namelist:
        os.rename((os.path.join(root,file)), (os.path.join(temptarget,file)))
    print("complete")
