split frozen_inference_graph.pb -b 49999999
cat x* > frozen_inference_graph.pb
