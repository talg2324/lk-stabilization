import numpy as np
import cv2
import utils
from tqdm import tqdm


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])

WINDOW_SIZE = 5        

def video_stabilization(input_video_path, output_video_path) -> None:
    """

    Stabilize the input video using homography
    1. Locate corners with Harris Detector
    2. Find matching feature points between the current frame and the reference frame using Lukas-Kanade
    3. Use RANSAC to filter outliers and compute the optimal affine transform
    4. Warp the current frame to the reference frame's coordinates
    5. Crop the edges and resize to aleviate border effects
    6. Occasionally update the reference frame and acquire new corners

    """

    cap, params = utils.get_video_parameters(input_video_path)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), params['fps'], (params['width'], params['height']), True)

    ret, im = cap.read()
    last_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    p0 = harris_detection(last_gray)
    out.write(im)

    h = np.eye(3)
    crop = 30 # crop and resize to fix border effects
    pbar = tqdm(total = params['frame_count'])
    i = 0

    while ret:
        ret, im = cap.read()
        pbar.update()

        if ret:
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            p1, h_n = lucas_kanade_optical_flow(p0, last_gray, im_gray, 10, 3)
            h = h @ h_n

            disp = im.copy()

            for px, py in p1.astype(np.int32):
                cv2.circle(disp, (px, py), 5, (0,0,255), -1)
            
            stable_im = cv2.warpPerspective(im, h, (im.shape[1], im.shape[0]), flags= cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            stable_im = cv2.resize(stable_im[crop:-crop, crop:-crop], (im.shape[1], im.shape[0]))
            out.write(stable_im)

            # disp = cv2.warpPerspective(disp, h, (im.shape[1], im.shape[0]), flags= cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            # disp = cv2.resize(disp, None, fx=0.5, fy=0.5)

            # cv2.imshow('frame', disp)
            # cv2.waitKey(10)

            last_gray = im_gray.copy()
            p0 = p1.copy()
            i += 1

            if i%10 == 0:
                p0 = harris_detection(last_gray)


    pbar.close()
    cap.release()
    out.release()

def lucas_kanade_optical_flow(p0, I1, I2, max_iter, num_levels):
    """
    Find the new set of points p1 corresponding to the input points p0.
    Return the new points and the motion model (h)

    1) Build pyramids for the input and reference image
    2) Init the motion model as the identity matrix
    3) Start at the bottom pyramid layer.  For each layer
        3a) Warp the input to reference coordinate
        3b) Find the optimal affine transform
        3c) Accumulate the warp and repeat for max_iter iterations
        3d) Scale the translation vector when raising the resolution
    """
    
    # Use powers of two for working with pyramids
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)

    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyramid_I2 = build_pyramid(I2, num_levels)

    h = np.eye(3)
    h_n = np.eye(3)
    max_levels = num_levels + 1
    p0_scaled = p0 / 2 ** num_levels

    for level in range(1, max_levels+1):
        I1_l = pyramid_I1[-level]
        I2_l = pyramid_I2[-level]
        I2_warp = cv2.warpPerspective(I2_l, h, (I2_l.shape[1], I2_l.shape[0]), flags= cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        for n in range(max_iter):
            h_n[:-1, :] = lk_step(p0_scaled, I1_l, I2_warp)
            h = h @ h_n
            I2_warp = cv2.warpPerspective(I2_l, h, (I2_l.shape[1], I2_l.shape[0]), flags= cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        if level < max_levels:
            h[0:2, 2] *= 2 # Scale the translation vector
            p0_scaled *= 2

    # Get the final output coordinates
    p1 = cv2.perspectiveTransform(np.expand_dims(p0, 0), h)
    return np.squeeze(p1), h

def lk_step(p0, I1, I2):
    """
    1)
    Solve the pixel-wise Flow Equation:
    (Ix Iy)@(u, v).T = -It     =>    Ad = b

    Using the Least Squares Solution
    d = inv(A.T @ A) @ A.T @ b

    2) Use RANSAC to compute an optimal affine transform
    """
    radius = WINDOW_SIZE//2
    
    # Flow equation
    Ix = cv2.Sobel(I2, cv2.CV_32F, 1, 0)
    Iy = cv2.Sobel(I2, cv2.CV_32F, 0, 1)
    It = cv2.subtract(I1, I2, dtype=cv2.CV_32F)
    AMat = np.stack((Ix, Iy), axis=-1)

    p1 = p0.copy()
    
    for i in range(p0.shape[0]):
        px, py = p0[i].astype(np.int32)

        A = AMat[py-radius:py+radius+1, px-radius:px+radius+1].reshape(-1, 2)
        At = A.T
        b = It[py-radius:py+radius+1, px-radius:px+radius+1].reshape(-1, 1)

        AtA = At @ A

        if np.linalg.det(AtA) > 0:
            d = np.linalg.inv(AtA) @ At @ b
            p1[i] = p0[i] + d.T

    h, inliers = cv2.estimateAffine2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=5, confidence=0.95)
    return h

def harris_detection(I1, k=4e-4, block_size=100):
    """
    
    Harris Corner detection
    Take the maximal corner from the response map within a block of size block_size
    
    Ignore smooth texture (low R values)
    """
    R = cv2.cornerHarris(I1, 2, 3, k=k)
    min_thresh = R.max() / 1e6
    points = []

    for i in range(0, R.shape[0], block_size):
        for j in range(0, R.shape[1], block_size):
            block = R[i:i+block_size, j:j+block_size]
            ii, jj = np.unravel_index(np.argmax(block), (block.shape[0], block.shape[1]))

            if block[ii, jj] > min_thresh:
                points.append([jj+j, ii+i])
            
    return np.stack(points).astype(np.float32)

def build_pyramid(image: np.ndarray, num_levels: int):
    """
    For each level in the pyramid:
    1) LPF
    2) Downsample

    Returns num_levels+1 levels (level 0 is the input image)
    """

    pyramid = [image.copy()]
    
    for l in range(num_levels):
        lpf = cv2.filter2D(pyramid[-1], None, PYRAMID_FILTER)
        pyramid.append(lpf[0:lpf.shape[0]:2, 0:lpf.shape[1]:2])

    return pyramid

def video_stabilization_simulation():
    """

    Stabilize the input video using homography
    1. Locate corners with Harris Detector
    2. Find matching feature points between the current frame and the reference frame using Lukas-Kanade
    3. Use RANSAC to filter outliers and compute the optimal homography (8-dof)
    4. Warp the current frame to the reference frame's coordinates
    5. Occasionally update the reference frame and acquire new corners

    """

    ret = True
    background = np.zeros((512,512, 3), dtype=np.uint8)
    for i in range(7):
        cv2.rectangle(background, (i*10,256-i*20), (i*20, 256+i*20), (255, 255, 255), -1)

    background = background + np.roll(np.flip(background, axis=1), -300, axis=1)
    txt = np.zeros_like(background)
    txt = np.hstack((txt,txt))
    cv2.putText(txt, 'Image (t)', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
    cv2.putText(txt, 'Stabilized Image (t)', (550, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

    last_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    p0 = harris_detection(last_gray, block_size=20)

    # Motion Model
    h_sim = np.eye(2,3)
    h_motion = np.zeros_like(h_sim)
    h_motion[:, -1] = (3, 0)
    h_noise = np.zeros_like(h_sim)

    # Reconstructed Motion
    h_rec = np.eye(3)
    t = 0

    while ret:

        # Simulation
        h_noise[:, -1] = np.random.standard_normal(2)*3
        h_motion[-1, -1] = 5 * np.sin(2 * np.pi * t / 10)
        h_sim += h_motion + h_noise
        im = cv2.warpAffine(background, h_sim, (background.shape[1], background.shape[0]))

        # Algorithm
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        p1, h_n = lucas_kanade_optical_flow(p0, last_gray, im_gray, 10, 3)
        h_rec =  h_rec @ h_n

        # draw the tracks
        for i, (new, old) in enumerate(zip(p1, p0)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(im, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
            cv2.circle(im, (int(a), int(b)), 5, (0,0,255), -1)

        rec_im = cv2.warpPerspective(im, h_rec, (background.shape[1], background.shape[0]), flags=cv2.WARP_INVERSE_MAP)
        im = np.hstack((im, rec_im))
        im[:, 512, :] = 255

        p0 = p1.copy()
        last_gray = im_gray.copy()

        cv2.imshow('frame', im+txt)
        cv2.waitKey(10)

        t += 1

        if t % 20 == 0:
            p0 = harris_detection(im_gray, block_size=20)
        elif t > 100:
            ret = False

def calc_mse(input_video_path):
    """
    
    Sanity check for stabilization quality.  
    Output video should have lower MSE than input.

    """
    input_cap, video_info = utils.get_video_parameters(input_video_path)
    frame_amount = video_info['frame_count']
    input_cap.grab()
    # extract first frame
    prev_frame = input_cap.retrieve()[1]
    # convert to greyscale
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mse = 0.0
    for i in range(1, frame_amount):
        input_cap.grab()
        frame = input_cap.retrieve()[1]  # grab next frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mse += ((frame - prev_frame) ** 2).mean()
        prev_frame = frame
    mean_mse = mse / (frame_amount - 1)
    return mean_mse

if __name__ == "__main__":
    video_stabilization_simulation()
