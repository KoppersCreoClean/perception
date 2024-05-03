# perception
Contains scripts and codes for perception tasks on the pan surface. 

Out of the current exising modules: creo_segmentation, block_detection and depth_sensing, creo-segmentation is in deployment. The other two are at a minimum viable stage and still in development.

There are also some test codes, configuration files and test images which help in testing, debugging and development.

For creo-segmentation:
1. Make changes in the segmentation_node.py main() to toggle between evaluation, testing and deployment modes. For evaluation purposes, it reads up images from the data directory. Simply make the suitable changes and run the python script.
2. The creo_segment is the main functional class code which is sourced in the segmentation_node to process the input image. Tune paramters here to improve the performance.
3. The performace_metrics.py is a utils file which has useful functions for evaluation of the segmentation performance. It is also sourced in segmentation_node and does not need to be altered unless when to add a new metric.
