Modelle können hier heruntergeladen werden: https://w-hs.sciebo.de/s/6UUXN3nf6hpJ3iJ?path=%2FSoftwareprojekte%2FUAV-II%2FModelle%2FDepthEstimation

Videos für Inference können hier heruntergeladen werden:https://w-hs.sciebo.de/s/6UUXN3nf6hpJ3iJ?path=%2FSoftwareprojekte%2FUAV-II%2FSoftwareprojekt%20Videos%2FDepthEstimation

Ein Docker-Container zur Ausführung des Neuronales Netzes zur Generierung von Tiefenbildern aus 2D-Farbbildern, zur Visualisierung der Netzausgabe und daraus generierter Punktwolken sowie für SLAM.



Der Container baut auf einer Kaskade anderer Container auf:

0. densedepth_ROSCudnn
Baut auf densedepthcudnn auf und fügt diesem noch die ROS-Integration mit RTABMAP-SLAM hinzu.

1. densedepthcudnn
Baut auf struct2depthCudnn auf und fügt QT-spezifische Inhalte, sowie Python-Inhalte hinzu, um Python basierte GUIs auf Python und QT-Basis zu verstehen.

2. struct2depthCudnn
Baut auf nvidia/cuda:10.1-devel auf. Beinhaltet vorallem weitere Cuda-spezifische Inhalte, sowie OpenGL-Schnittstellen zu schnellen Netzinferenz auf Cuda-fähigen Nvidia-Grafikkarten. 



Bauen des Containers:

Die oben genannten Image-Hierarche muss berücksichtigt und die jeweiligen build.sh-Skripte der Reihe nach ausgeführt werden.
Hinweis: Obwohl es sich um Dockercontainer handelt, sind CUDA- und Nvidia-spezifische Abähngigkeiten zu beachten:
In der Container-Umgebung werden Nvidia-spezifische Inhalte installiert, wozu auf der Rechner-Plattform die Hardware mit entsprechenden Treiber installiert sein müssen.



Starten des Containers:

1) Öffne Terminal und starte ./run.sh

2.) Wechsel in das DenseDepth-Verzeichnis, in dem sich auch die DepthInferenceAndVisualization.py befindet. Das Verzeichnis kann in der run.sh eingestellt werden.
(Bei mir in: /home/marc/DepthInferenceAndVisualization)

3.) Lade die .bashrc mit:
source /home/marc/.bashrc

4.) Lade die setup.bash des Catkin-WS mit:
source ../catkin_build_ws/devel/setup.bash

5.) Öffne ein zweites Terminal und öffne hier eine interaktive Umgebung zum nun laufenden Container:
docker exec -it densedepth_roscudnn /bin/bash

6.) Lade in Terminal 2 die .bashrc mit:
source /home/marc/.bashrc

7.) Starte in Terminal 2 einen Roscore

8.) In Terminal 1 führe aus: python3 demo.py

docker exec -it densedepth_roscudnn /bin/bash


Falls RTABMAP gestartet werden soll:

1.) Starte eine neue interaktive Docker-Shell:
docker exec -it densedepth_roscudnn /bin/bash
source /home/marc/.bashrc

2.) Starte ZWEI neue interaktive Docker-Shell:
docker exec -it densedepth_roscudnn /bin/bash
source /home/marc/.bashrc

3.) Starte die TFs:
rosrun tf static_transform_publisher 0 0 0 -1.57079632679489 0 -1.5707963267948966 base_link camera_link 100

rosrun tf static_transform_publisher 0 0 0 -1.57079632679489 0 -1.5707963267948966 base_link depth_link 100

4.) Starte eine neue interaktive Docker-Shell:
docker exec -it densedepth_roscudnn /bin/bash
source /home/marc/.bashrc

5.) Starte den RGBD-Knoten von RTABMAP_ROS:
rosrun rtabmap_ros rgbd_odometry rgb/image:=/camera/rgb/image_rect_color depth/image:=/cama/depth_registered/image_raw rgb/camera_info:=/camera/rgb/camera_info

6.)

Starte RTABMAP mit:
roslaunch rtabmap_ros rgbd_mapping.launch 

Besser:
roslaunch rtabmap_ros rgbd_mapping.launch rtabmap_args:="--Odom/ResetCountdown 30 --Odom/FilteringStrategy 1 Optimizer/Strategy 2 --Marker/VarianceLinear 0.04 --Marker/VarianceAngular 0.4 --Marker/MaxDepthError 0.2 --GridGlobal/OccupancyThr 0.8 --Grid/RangeMin 0.08 --Grid/CellSize 0.2 --Vis/FeatureType 8 --Mem/IntermediateNodeDataKept true --Vis/EstimationType 1 --Odom/Strategy 0 --OdomF2M/BundleAdjustment 3 --OdomF2M/BundleAdjustmentMaxFrames 60 --OdomF2M/ScanRange 30 --Vis/ForwardEstOnly true --Vis/RefineIterations 10 --Vis/InlierDistance 0.05 --Vis/EpiploarGeometryVar 0.2 --Vis/MinInliers 20 --Vis/MinDepth 0.8 --Vis/BundleAdjustment 3" 


bzw. mit anderen Parametrierungen zB.:

roslaunch rtabmap_ros rgbd_mapping.launch rtabmap_args:="--Odom/ResetCountdown 30 --Odom/FilteringStrategy 1 Optimizer/Strategy 3 --Marker/VarianceLinear 0.04 --Marker/VarianceAngular 0.4 --Marker/MaxDepthError 0.4 --GridGlobal/OccupancyThr 0.8 --Grid/RangeMin 0.08 --Grid/CellSize 0.2 --Vis/FeatureType 8

roslaunch rtabmap_ros rgbd_mapping.launch rtabmap_args:="--Odom/ResetCountdown 30 --Odom/FilteringStrategy 1 Optimizer/Strategy 3 --Marker/VarianceLinear 0.04 --Marker/VarianceAngular 0.4 --Marker/MaxDepthError 0.4 --GridGlobal/OccupancyThr 0.8 --Grid/RangeMin 0.08 --Grid/CellSize 0.2 --Vis/FeatureType 8 --Mem/STMSize 30 --Mem/IntermediateNodeDataKept true" 


roslaunch rtabmap_ros rgbd_mapping.launch rtabmap_args:="--Odom/ResetCountdown 30 --Odom/FilteringStrategy 1 Optimizer/Strategy 3 --Marker/VarianceLinear 0.04 --Marker/VarianceAngular 0.4 --Marker/MaxDepthError 0.4 --GridGlobal/OccupancyThr 0.8 --Grid/RangeMin 0.08 --Grid/CellSize 0.2 --Vis/FeatureType 8 --Mem/STMSize 30 --Mem/IntermediateNodeDataKept true --Vis/EstimationType 2 --Mem/DepthAsMask false" 


roslaunch rtabmap_ros rgbd_mapping.launch rtabmap_args:="--Odom/ResetCountdown 30 --Odom/FilteringStrategy 1 Optimizer/Strategy 2 --Marker/VarianceLinear 0.04 --Marker/VarianceAngular 0.4 --Marker/MaxDepthError 0.2 --GridGlobal/OccupancyThr 0.8 --Grid/RangeMin 0.08 --Grid/CellSize 0.2 --Vis/FeatureType 8 --Mem/IntermediateNodeDataKept true --Vis/EstimationType 1 --Odom/Strategy 0 --OdomF2M/BundleAdjustment 3 --OdomF2M/BundleAdjustmentMaxFrames 60 --OdomF2M/ScanRange 30 --Vis/ForwardEstOnly true --Vis/RefineIterations 10 --Vis/InlierDistance 0.05 --Vis/EpiploarGeometryVar 0.2 --Vis/MinInliers 20 --Vis/MinDepth 0.8 --Vis/BundleAdjustment 3" 


Zu DepthInferenceAndVisualization.py
Ermöglicht die Inferenz eines NN zur Tiefenbilderstellung und published RGB und Tiefenbild über ROS. Darüber hinaus wird eine GUI gestartet, die RGB und Tiefenbild sowie eine 3D-Punktwolke darstellt.

Umgesetzt werden folgende Features:
1) LADEN eines Modells (hdf5, h5) über einen File Explorer- Dialog
2) LADEN eines Videos oder eines Bildes mit einen File-Explorer-Dialog
Aktuelle Formate: 
Für Bilder *.png, *.ppm
Für Videos: *.avi

3) INFERENZ mit den Eingabebildern zur Generierung von Tiefenbildern mit dem geladenen Modell
4) VISUALISIEREN von Tiefenbild und GENERIERUNG einer 3D-Punktwolke mit entsprechender Visualisierung
Dem Interpreter können Parameter für das Programm übergeben werden, oder die Parameter werden direkt im Quellcode eingestellt.
Über diese kann eingestellt werden, welcher VISUALSIERUNGSMODUS verwendet wird:
-RGB und Tiefenbild
-RGB und Punktwolke
-RGB, Tiefenbild und Punktwolke (Standard)

Zu beachten: Das Programm diente uns zu Testzwecken und stellte eigentlich kein Produkt im Sinne der UAV-II-Anforderungen dar.
Jedoch stellte sich heraus, dass es sehr hilfreich sein kann, Tiefenbilder und 3D-Punktwolken im Zuge der NN-Inferenz zu visualisieren.
In der Praxis ergeben sich schnell Anforderungen zur Verbesserung des Handlings mit der Netzausgabe. Zum Beispiel möchte man neben einfachen Bildern
zusätzlich Videos als Eingabe bestimmen. Da - je nach Netz und Settings- die resultierende FPS Zahl im niedrigen einstelligen Bereich liegen kann ist eine Skip-Funktion wünschenswert,
mit der man im Video schneller an eine interessante Stelle gelangen kann. Aber auch weitere Funktionen, wie eine Pause und Resume-Logik wären wünschenswert, die aktuell nicht im Tool implementiert sind.

Dieses Tool kann ausgebaut werden um im Idealfall einen Frontend für beliebige Neuronale Netze zu dienen.  
