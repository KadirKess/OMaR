# Prototype
> This part of the project is dedicated to the prototyping step. Implementation is in python in this section


## Preprocessing

### staff detection:
First implementation was using shortest path to detect the lines but too costly and complicated to use. It has pros in the line detection when it is not straight but since a deskew is made it's not relevant. Hence a projection method is used for better time performance and code simplicity.