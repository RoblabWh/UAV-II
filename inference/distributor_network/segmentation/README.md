# Segmentierung
## Getting Started
Um die Ergebnisse dieses Programms nachzustellen wird das Betriebssystem Ubuntu 18.04 benötigt. Das Programm wurde in Python 3.6 getestet.
## Dependencies

```
pip3
torch>=0.4.0
```
Zum installieren der benöitgten Packete den Befehl `sudo -H pip3 install -r requirements.txt`

## Running Programm

Diese Semantische Segmentierung unterstützt zwei Datensätze:
1.) Pascal VOC
2.) Pascal Person Part

Zum Ausführen der Programme folgende Befehle eingeben:

```
- python3 voc_new.py
- python3 personpart.py
```