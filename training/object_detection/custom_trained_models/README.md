### Aufteilung des Graphen in mehrere Dateien

Befehl: ```split frozen_inference_graph.pb -b 24999999```

Dies ermöglicht größere Dateien (>100MB) in mehrere kleine aufzuteilen.


### Zusammenfügen der Dateien

Befehl: ```cat x* > frozen_inference_graph.pb```
