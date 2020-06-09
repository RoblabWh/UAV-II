# Network Interface
## Einleitung
Das Network Interface enthält das Programm für die Kommunikationsschnittstelle vom UAV-Network zum Distributor-Network und sendet bei erstellter Verbindung Periodisch den Videostream der Drohne in das Distributor Netzwerk. Zudem können Daten zur Unterstützung des Operators an dieser stelle zugesand werden

## Modulbeschreibung
Die Datei *network.py* wird von der Drohne genutzt, um eine Verbindung zu dem Distributor-Network aufzubauen

Das Modul *utils.py* bietet eine Schnittstelle zum schnellen erzeugen einer Socket-Verbindung im Distributor-Netzwerk.
