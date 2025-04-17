# import cv2
# import math
# import numpy as np
# import sys
# def apply_mask(matrix, mask, fill_value):
#     # print("MATRIX=", matrix)
#     # print("mask=\n" ,mask)
#     # print("fill value=\n", fill_value)
                 
#     masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
#     print('MASKED=', masked)
#     return masked.filled()


# def apply_threshold(matrix, low_value=255, high_value=255):
#     low_mask = matrix < low_value
#     print("low mask=", low_mask)
    
#     matrix = apply_mask(matrix, low_mask, low_value)
#     print('Low MASK->', low_mask, '\nMatrix->', matrix)

#     high_mask = matrix > high_value
#     matrix = apply_mask(matrix, high_mask, high_value)

#     return matrix


# def simplest_cb(img, percent):
#     assert img.shape[2] == 3
#     assert percent > 0 and percent < 100
#     print("shape of image = ", img.shape[2])

#     half_percent = percent / 200.0
#     print('HALF PERCENT->', half_percent)

#     channels = cv2.split(img)
#     print('Channels->\n', channels)
#     print('Shape->', channels[0].shape)
#     print('Shape of channels->', len(channels[2]))

#     out_channels = []
#     for channel in channels:
#         assert len(channel.shape) == 2

# 	# find the low and high precentile values (based on the input percentile)
#         height, width = channel.shape
#         vec_size = width * height
#         flat = channel.reshape(vec_size)
#         print('vec=', vec_size, '\nFlat=', flat)
#         assert len(flat.shape) == 1

#         flat = np.sort(flat)

#         n_cols = flat.shape[0]
#         print("Number of columns = ", n_cols)

#         low_val = flat[math.floor(n_cols * half_percent)]
#         high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

#         print("Lowval: ", low_val)
#         print("Highval: ", high_val)
#         print(flat[60])
#         print(flat[11940])

#         # saturate below the low percentile and above the high percentile
#         thresholded = apply_threshold(channel, low_val, high_val)
#         # scale the channel
#         normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
#         out_channels.append(normalized)

#     return cv2.merge(out_channels)


# if __name__ == '__main__':
#     # img = cv2.imread(sys.argv[1])
# 	# cap = cv2.VideoCapture('/media/dheeraj/9A26F0CB26F0AA01/WORK/github_repo/Dehazing/haze-videos/Whale.mov')
# 	cap = cv2.VideoCapture('haze-videos/dolphin.mp4')
#    	# img = cv2.imread('/home/dheeraj/Downloads/Whale.mov')
# 	while True:
	
# 		ret, frame = cap.read()
	
# 		out = simplest_cb(frame, 1)
# 		cv2.imshow("Before", frame) 
# 		cv2.imshow("After", out)
# 		cv2.waitKey(0)
# cap.release()
# cv2.destroyAllWindows()

#######################################################################################################

# import streamlit as st
# import cv2
# import numpy as np
# import tempfile
# import math


# def apply_mask(matrix, mask, fill_value):
#     masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
#     return masked.filled()


# def apply_threshold(matrix, low_value=255, high_value=255):
#     low_mask = matrix < low_value
#     matrix = apply_mask(matrix, low_mask, low_value)
#     high_mask = matrix > high_value
#     matrix = apply_mask(matrix, high_mask, high_value)
#     return matrix


# def simplest_cb(img, percent):
#     assert img.shape[2] == 3
#     assert 0 < percent < 100
#     half_percent = percent / 200.0
#     channels = cv2.split(img)
#     out_channels = []
#     for channel in channels:
#         height, width = channel.shape
#         vec_size = width * height
#         flat = channel.reshape(vec_size)
#         flat = np.sort(flat)
#         n_cols = flat.shape[0]
#         low_val = flat[math.floor(n_cols * half_percent)]
#         high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]
#         thresholded = apply_threshold(channel, low_val, high_val)
#         normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
#         out_channels.append(normalized)
#     return cv2.merge(out_channels)


# def process_video(input_path, output_path):
#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         dehazed_frame = simplest_cb(frame, 1)
#         out.write(dehazed_frame)
    
#     cap.release()
#     out.release()


# def main():
#     st.title("Video Dehazing Web App")
#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
#             temp_input.write(uploaded_file.read())
#             input_path = temp_input.name
        
#         output_path = input_path.replace(".mp4", "_dehazed.mp4")
#         st.text("Processing video...")
#         process_video(input_path, output_path)
#         st.video(output_path)


# if __name__ == "__main__":
#     main()

######################################################################################

# import streamlit as st
# import cv2
# import numpy as np
# import tempfile
# import math


# def apply_mask(matrix, mask, fill_value):
#     masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
#     return masked.filled()


# def apply_threshold(matrix, low_value=255, high_value=255):
#     low_mask = matrix < low_value
#     matrix = apply_mask(matrix, low_mask, low_value)
#     high_mask = matrix > high_value
#     matrix = apply_mask(matrix, high_mask, high_value)
#     return matrix


# def simplest_cb(img, percent):
#     assert img.shape[2] == 3
#     assert 0 < percent < 100
#     half_percent = percent / 200.0
#     channels = cv2.split(img)
#     out_channels = []
#     for channel in channels:
#         height, width = channel.shape
#         vec_size = width * height
#         flat = channel.reshape(vec_size)
#         flat = np.sort(flat)
#         n_cols = flat.shape[0]
#         low_val = flat[math.floor(n_cols * half_percent)]
#         high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]
#         thresholded = apply_threshold(channel, low_val, high_val)
#         normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
#         out_channels.append(normalized)
#     return cv2.merge(out_channels)


# def process_video(input_path, output_path):
#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         dehazed_frame = simplest_cb(frame, 1)
#         out.write(dehazed_frame)
    
#     cap.release()
#     out.release()


# def main():
#     st.title("Video Dehazing Web App")
#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
#             temp_input.write(uploaded_file.read())
#             input_path = temp_input.name
        
#         output_path = input_path.replace(".mp4", "_dehazed.mp4")
        
#         st.subheader("Original Video")
#         st.video(input_path)
        
#         st.text("Processing video...")
#         process_video(input_path, output_path)
        
#         st.subheader("Dehazed Video")
#         st.video(output_path)


# if __name__ == "__main__":
#     main()

#####################################################################################################

import streamlit as st
import cv2
import numpy as np
import tempfile
import math
import shutil
import os
# import time


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value=255, high_value=255):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)
    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)
    return matrix


def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert 0 < percent < 100
    half_percent = percent / 200.0
    channels = cv2.split(img)
    out_channels = []
    for channel in channels:
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        flat = np.sort(flat)
        n_cols = flat.shape[0]
        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]
        thresholded = apply_threshold(channel, low_val, high_val)
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)
    return cv2.merge(out_channels)

# def process_video(input_path, output_path):
#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         dehazed_frame = simplest_cb(frame, 1)
#         out.write(dehazed_frame)
    
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Ensure H.264 encoding
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dehazed_frame = simplest_cb(frame, 1)
        out.write(dehazed_frame)

    cap.release()
    out.release()  # Ensure file is properly written


def main():
    st.title("Video Dehazing Web App")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_file.read())
            input_path = temp_input.name
        # temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        # temp_input.write(uploaded_file.read())
        # temp_input.close()
        # input_path = temp_input.name
        
        output_path = input_path.replace(".mp4", "_dehazed.mp4")
        
        st.subheader("Original Video")
        st.video(input_path, format='video/mp4')
        
        st.text("Processing video...")
        process_video(input_path, output_path)
        
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        shutil.copy(output_path, temp_output.name)
        
        st.subheader("Dehazed Video")
        st.video(temp_output.name, format='video/mp4')
        
        os.unlink(output_path)  # Cleanup after processing
        os.unlink(temp_output.name)


        # # Delay to allow Streamlit to finish using the file
        # time.sleep(1)

        # # Ensure the file is not in use before deleting
        # try:
        #     os.remove(temp_output.name)
        # except PermissionError:
        #     st.warning("Could not delete temporary file immediately. It will be removed after the session.")

if __name__ == "__main__":
    main()
