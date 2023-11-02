from ultralytics import FastSAM
from PIL import Image
import matplotlib.pyplot as plt

# Define an inference source
source = 'C:/Users/ivanin.em/Desktop/sam/frame_200.jpg'

# Create a FastSAM model
model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt

# Run inference on an image
everything_results = model(source, device='gpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# Show the results
for r in everything_results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    # im.save('frame.jpg')  # save image