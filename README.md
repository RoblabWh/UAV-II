# UAV II - Einsatz von KI im Gefahrenbereich
## Problemstellung
* Könnte die Drohne bei Konzentrationsverlust des Operators mithilfe eines Deep Learning Verfahrens selbstständig navigieren?
* Könnten Gegenstände und Personen im Gefahrenbereich dann auch mithilfe von Deep Learning automatisch erkannt werden?
* Könnte räumliche Tiefe mithilfe von Deep Learning gewonnen werden, auch wenn keine Sensorik zur Entfernungsmessung zur Verfügung steht?

## Idee und Konzept
* Einsatz eines CNNs als Navigationshilfe für den Drohnenoperator
* Objekterkennung zur Detektion von Warnschildern
* Segmentierung zur Erkennung von Personen und Unfallauslösern (z.B. Gasflasche)
* Tiefenbildeinschätzung als Sensorikersatz
* Aufbau von Kartenmodellen unter Einsatz neuronaler Netze

## Technische Umsetzung 
* Implementierung und Training eines neuronalen Faltungsnetzes (CNN) mittels Keras und ca. 90.000 aufgenommenen Bildern zur relativen Lokalisierung und Navigation in Korridoren 
* Automatisierte Generierung von neuen Datensätzen für die Objekterkennung mithilfe von OpenCV
Datensätze der semantischen Segmentierung: PascalVoc und PersonPart
* Modifikation und Training eines neuronalen Netzes zur Tiefenschätzung in Räumlichkeiten des Gefahrenbereichs zur Bereitstellung einer lokalen 3D-Ansicht
* Segmentierung von Korridorböden mit DeepLab-ResNet und Fusion zu einer 2D-Karte mit ORB2-SLAM zur Trajektorienbestimmung
* Evaluierung verschiedener neuronaler Netze (SSD, RFCN, Struct2Depth, DenseDepth, CNNDepth, RefineNet, RefineNet Light,ResNet)
