# robot_yolo

This code uses the YOLOv3 object detection model to detect and classify living and non-living objects in a video

## Output:

The code will print the x1, x2, and label for each living and non-living object detected. The x1 and x2 values are the coordinates of the bottom-left and bottom-right corners of the bounding box, respectively. The label is either 'living' or 'nonliving'.

### Example output:
```
100 200 living
300 400 nonliving
```
## Notes:

The code uses a threshold of 0.5 for both the confidence and NMS scores. You can adjust these thresholds to improve the accuracy or speed of the detection.
The code currently only detects a subset of living and non-living objects. You can add more objects to the living and nonliving lists to improve the detection performance.
