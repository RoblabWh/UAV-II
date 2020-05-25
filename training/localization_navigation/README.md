# Training und Testing
Dieses Verzeichnis beinhaltet unter anderem zwei Jupyter Notebooks die zum Training und Testen des CNN verwendet werden können. Die Jupyter Notebooks sind dokumentiert und selbsterklärend. Das Jupyter Notebook wurde zum Testen als befriedigend eingestuft, da die realen Tests aussagräftiger waren. Dabei konnten z.B. Artefakte wie rauschen und reflektionen mit berücksichtigt werden.  
## Training
Der Ordner "checkpoints" beinhaltet das gespeicherte Model, welches beim Training abgespeichert wurde.
Die Trainings- und Evaluierungsbilder befinden sich in dem "dataset" Ordner.
In dem Ordner "plots" befindet sich ein Bild mit einem Graphen, der die Accuracy und Loss der verschiedenen Models auflistet.
Unter anderem befindet sich in dem Ordner auch ein detailreiches Bild von der Architektur des CNN. 
## Testing
Der Ordner "checkpoints" beinhaltet das gespeicherte Model, welches zum endgültigen Testen verwendet wurde.
Bei dem Testen, können die Heatmeaps verschiedener Layer angezeigt werden. Zudem können die Aktivierungsfunktionen ausgetauscht werden, um das Verhalten des CNN zu verändern. Diese Änderung an der Architektur des CNN wird in dem "tmp" Ordner abgespeichert.
