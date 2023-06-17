import imageio as io
import matplotlib.pyplot as plt
import time

camera = io.get_reader('<video0>')
meta = camera.get_meta_data()
delay = 1 / meta['fps']

for frame_counter in range(15):
    frame = camera.get_next_data()
    time.sleep(delay)
    plt.imshow(frame)
    plt.show()

camera.close()

