import numpy as np

#load binary array/matrix
a = np.load("./var0.npy")

# create txt file to write
f = open("var0.txt", "w")

# write 5x5x1x32
for kernel in range(0,32):
    for channel in range(0,1):
        for row in range(0,5):
            for col in range(0,5):
                f.write(str(a[row][col][channel][kernel]))
                f.write("\n")
f.close()

#####################################################
a = np.load("./var1.npy")
f = open("var1.txt", "w")

# write 32
for kernel in range(0,32):
    f.write(str(a[kernel]))
    f.write("\n")
f.close()

#####################################################
a = np.load("./var2.npy")
f = open("var2.txt", "w")

# write 5x5x32x64
for kernel in range(0,64):
    for channel in range(0,32):
        for row in range(0,5):
            for col in range(0,5):
                f.write(str(a[row][col][channel][kernel]))
                f.write("\n")
f.close()

####################################################
a = np.load("./var3.npy")
f = open("var3.txt", "w")

# write 64
for kernel in range(0,64):
    f.write(str(a[kernel]))
    f.write("\n")
f.close()

###################################################
a = np.load("./var4.npy")
f = open("var4.txt", "w")

# write 3136x1024
for kernel in range(0,3136):
    for channel in range(0,1024):
        f.write(str(a[kernel][channel]))
        f.write("\n")
f.close()

###################################################
a = np.load("./var5.npy")
f = open("var5.txt", "w")

# write 1024
for kernel in range(0,1024):
    f.write(str(a[kernel]))
    f.write("\n")
f.close()

###################################################
a = np.load("./var6.npy")
f = open("var6.txt", "w")

# write 1024x10
for kernel in range(0,1024):
    for col in range(0,10):
        f.write(str(a[kernel][col]))
        f.write("\n")
f.close()

###################################################
a = np.load("./var7.npy")
f = open("var7.txt", "w")

# write 10
for kernel in range(0,10):
    f.write(str(a[kernel]))
    f.write("\n")
f.close()
