import pyrealsense2 as rs
import numpy as np
import cv2
pipe = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipe.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)
#cv2.namedWindow('win1', cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow('win2', cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow('win3', cv2.WINDOW_AUTOSIZE)
count = 0
try:
    while True:
        frames = pipe.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()


        if not depth_frame or not color_frame:
            continue
        print('Frame ', count)
        count = count + 1
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #color_image = cv2.fastNlMeansDenoisingColored(color_image, None, 10, 10, 7, 15)
        #color_image = cv2.fastNlMeansDenoisingColored(color_image,None,10,10,7,21)
        gray_color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        #gray_color_image = gray_color_image * 0.75
        gray_color_image2 = cv2.convertScaleAbs(gray_color_image,alpha=0.25, beta=0)
        height, width = gray_color_image.shape
        #print('RGB', color_image)
        hls_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HLS)
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        #print('HLS', hls_image)
        lower = np.array([0, 175, 0])
        upper = np.array([180, 200, 255])
        mask = cv2.inRange(hls_image, lower, upper)
        #mask = cv2.bitwise_not(mask)

        mask_gray_color_image = np.bitwise_and(gray_color_image2, mask)
        #mask = cv2.fastNlMeansDenoising(gray_color_image, None, 9, 13)
        #gaussBlur_mask = cv2.GaussianBlur(mask_gray_color_image, (7, 7), 0)
        mask = cv2.fastNlMeansDenoising(mask, None, 9, 13)
        edges = cv2.Canny(mask, 200, 400)
        
        mask = np.zeros_like(edges)
        height, width = edges.shape
        polygon = np.array([[
            (0, height / 2),
            (width, height / 2),
            (width, height),
            (0, height)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)

        #canny_mask = canny_mask[height // 2: height, 0:width]

        #mask = cv2.inRange(gray_color_image, 200, 255)
        #print(width, height, mask.shape)
        #mask = mask[height // 2: height, 0:width]
        #print(mask.shape)
        #gaussBlur_mask = cv2.GaussianBlur(mask, (7, 7), 0)
        #canny_mask = cv2.Canny(gaussBlur_mask, 100, 200, apertureSize = 5)
        lines =	cv2.HoughLinesP(cropped_edges, 
                                1, 
                                np.pi / 180, 
                                10, 
                                np.array([]), 
                                minLineLength=8, maxLineGap=4)
        #print(width, height, canny_mask.shape)
        if type(lines) != type(None):
            print("find ",len(lines), "lines")
            for line in lines:
                print(line)
                x1,y1,x2,y2 = line[0]
                if x2 == x1:
                    continue
                weight = (y2 - y1) / (x2 - x1)
                bias   = (y2 * x1 - y1 * x2) / (x1 - x2)
                print(weight, bias)
                '''
                nX1, nY1, nX2, nY2 = x1, y1, x2, y2
                i = 0
                while(1):
                    nX1 = int(nX1 + i * 10)
                    nY1 = int(weight * nX1 + bias)
                    i = i + 1
                    print(nX1, nY1)
                    if nX1 >= width or nY1 < 0 or nY1 >= height / 2 or canny_mask[nY1, nX1] == 0:
                        x1 = nX1 + 10
                        y1 = int(weight * x1 + bias)
                        break

                i = 0
                while(1):
                    nX2 = int(nX2 - i * 10)
                    nY2 = int(weight * nX2 + bias)
                    i = i + 1
                    print(nX2, nY2)
                    if nX2 < 0 or nY2 < 0 or nY2 >= height / 2 or canny_mask[nY2, nX2] == 0:
                        x2 = nX2 + 10
                        y2 = int(weight * x2 + bias)
                        break
                '''
                cv2.line(color_image,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow('win1', color_image)
        #cv2.imshow('win2', gray_color_image)
        #cv2.imshow('win3', gray_color_image2)
        #cv2.imshow('win4', hls_image)
        #cv2.imshow('win5', hsv_image)
        cv2.imshow('win6', mask_gray_color_image)
        #cv2.imshow('win7', gaussBlur_mask)
        cv2.imshow('win8', cropped_edges)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipe.stop()