import os

path = 'D:/UnderWaterResearch/Codes/Semi-UIR/data/test/Seathru2K_D5/'

def getfiles():
    filenames=os.listdir('D:/UnderWaterResearch/Codes/Semi-UIR/data/test/Seathru2K_D5/input')
    print(filenames)
    return filenames



if __name__ == '__main__':

    if not os.path.exists(path + 'test.txt'):
        # os.mknod('test.txt')
        open(path + 'test.txt', 'w')
        print('OPEN ')

    a = getfiles()
    # a.spilt('')
    l = len(a)
    with open(path + "test.txt", "w") as f:
        for i in range(l):
            print(a[i])
            x = a[i]
            f.write(x)
            f.write('\n')
        f.close()
        # f.flush(0)

    print()