# Split large files
split frozen_inference_graph.pb -b 24999999

# Merge them together 
cat x* > frozen_inference_graph.pb
