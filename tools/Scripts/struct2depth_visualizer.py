#====================================
#Modulename: struct2depth_visualizer
#Author Artur Leinweber
#Email: arturleinweber@live.de
#====================================

#Imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sys
from sklearn.preprocessing import minmax_scale


def main():
   
   normalize = False

	#Checking Paramters
   if len(sys.argv) >= 2:
      if (sys.argv[1])[-4:] == ".npy":
         path_to_npy = sys.argv[1]
         if (len(sys.argv) == 3):
            if(sys.argv[2] == "n"):
                normalize = True
   else:
      print("Need one argument for npy datapath!")
      exit(0)

	#Loading numpy data
   loaded_npy = np.load(path_to_npy)
   print(type(loaded_npy))
   print(loaded_npy.shape)
	
	#Loading Dimensions
   y_len = loaded_npy.shape[0]
   x_len = loaded_npy.shape[1]

	#Generating numpy arrays for X-,Y- and Z- Axis
   x_array = np.linspace(0, x_len, x_len)
   y_array = np.linspace(0, y_len, y_len)
   #z_array = np.zeros((y_len, x_len))
   z_array = []

	#Generating a Meshgrid
   X_array, Y_array = np.meshgrid(x_array, y_array)

	#Reshape Array for Z-Axis
   for i in range(y_len):
      for j in range(x_len):
         #z_array.append(float(loaded_npy[i][j][0]))
         z_array[i][j] = float(loaded_npy[i][j][0])

	#Normalization
   if (normalize == True):
       z_array = minmax_scale(z_array)
    
   print(z_array.shape)
	
	#Plotting figure
   
   fig = plt.figure(path_to_npy)
   ax = plt.axes(projection='3d')
   surf = ax.plot_surface(X_array, Y_array, z_array,cmap=cm.coolwarm,antialiased=False)
   
   #fig = plt.figure()
   #ax = fig.add_subplot(111, projection='3d')
   #ax.plot(x_array, y_array, z_array, 'r+')
   #ax.plot_surface(X_array, Y_array, z_array, cmap=cm.coolwarm,antialiased=False)
   ax.axis('equal')
   ax.set_xlabel('x')
   ax.set_ylabel('y')
   ax.set_zlabel('z')

   #ax.view_init(100, 10)
   #fig.colorbar(surf, shrink=0.5, aspect=5)
   plt.show()


if __name__== "__main__":
	main()
