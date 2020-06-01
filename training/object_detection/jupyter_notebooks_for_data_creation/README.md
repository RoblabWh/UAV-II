## CreatingDatasetImages

Die Warnzeichen (.png Dateien) wurden zunächst in Ordnern gespeichert.  
Bsp.
```
|- Harmful 
    |- Harmful01.png  
    |- Harmful02.png
|- Poison
    |- Poison02.png
    |- Poison03.png
```
Nachdem der Pfad zu den Warnzeichen angegeben ist und die Daten der Hintergründe geladen werden können, wird in einer Loop der Datensatz generiert. Die Bilder werden rotiert, getrübt und gefärbt und anschließend neu gespeichert.

**CreatingDatasetImages ersetzt das CreatingImages, welches zurvor zum Erzeugen des Datensatzes verwendet worden ist.**

## ImageDataToJson

In diesem Notebook können einzelne Hintergründe geladen werden, um die Punkte u.a. erneut abzuspeichern. Existiert der Hintergrund bereits in background.json, so wird auf dieses Json-Objekt zugegriffen und die alten Daten überschrieben.
