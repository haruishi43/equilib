# Pano2Perspective
Convert panorama image to perspective image

[Installation](https://github.com/Toraudonn/pano2perspective/#installation)

[Usage](https://github.com/Toraudonn/pano2perspective/#usage)

## Dependencies:

- Ubuntu 16.04
- Python (>3.5)
- Numpy
- Pybind11 (submodule of this repo)
- OpenCV
- Eigen
- CUDA (=9.0)

### Optional Dependencies:

- OpenMP (for cpu)
- matplotlib, opencv-python (for running test)

## Installation

1. Install [CUDA](https://developer.nvidia.com/cuda-90-download-archive)

```
nvcc --version
```

This should print out that your CUDA version is `9.0`.

1. Install Python dependencies:

```
sudo apt-get install -y python3 python3-pip python3-dev python3-venv python3-numpy
pip3 install --upgrade pip

sudo apt-get install libopencv-dev libeigen3-dev
```

1. Install OpenCV for Python

`pip3 install opencv-python`

Or manually build and link it against the Python you are using by doing [this](https://www.learnopencv.com/install-opencv3-on-ubuntu/).

1. Install Eigen:

```
sudo apt-get install -y libxmu-dev libxi-dev
cd $HOME
wget http://bitbucket.org/eigen/eigen/get/3.2.10.zip
unzip 3.2.10.zip && rm 3.2.10.zip
cd eigen-eigen-b9cd8366d4e8
mkdir build; cd build
cmake ..
sudo make install
```
You may have to link Eigen against local:

```
cd /usr/local/include/
sudo ln -sf eigen3/Eigen Eigen
```

1. Clone this repo:

```
git clone --recurse https://github.com/Toraudonn/pano2perspective.git
```

1. Build the library:

```
python setup.py build
python setup.py install
```

1. Check if it installed correctly:

Open Python prompt:

```
>>> import pano2perspective
>>> help(pano2perspective)
```

This should output all of the classes and functions in this library.

## Usage

Use the `help()` function to details about this library:

```
>>> import pano2perspective
>>> help(pano2perspective)
```

### Single image

For converting a single image.
You can choose whether or not to use the GPU.
If you choose to not run it using cuda, the code runs on OpenMP for parallel execution.
If you don't have OpenMP, it will run concurrently.
On a multi-core processor, OpenMP execution may run faster than cuda.

```Python
import pano2perspective

# Read panorama image as numpy array
pano = cv2.imread("pano.jpg")

# split numpy array into channels and put it in a list
pano_arr = [pano[:,:, 0],  # R
            pano[:, :, 1], # G
            pano[:, :, 2]] # B

# Create a list with yaw, pitch, roll
# forward = [0, 0, 0]
# yaw: -pi < b < pi
# pitch: -pi/2 < a < pi/2
rot = [0, 0, 0]  # [yaw, pitch, roll]

pers = np.array(
        pano2perspective.process_single_image(pano_arr, # input panorama
                                            rot,  # rotation
                                            True, # use cuda
                                            480,  # width
                                            360,  # height
                                            90.0) # FOV
        copy=False)

cv2.imshow("output", dst_img)
```

Coverting via passing center point in panorama is not supported in this function.

In later updates, this function will become deprecated.

### Multiple images (video)

For multiple images (such as videos or files of images), using GPU is more than 50 times faster.
Use the `Pano2Perspective` class for these kind of tasks:

```Python
import pano2perspective as pano

# Initialize the class with cropping size (width, height)
p2p = pano.Pano2Perspective(480, 360, 90.0) # intialize using perspective image size and fov of width
p2p.cuda(0)  # use cuda and set device id

cap = cv2.VideoCapture("path_to_video")
while(cap.isOpened()):
    _, frame = cap.read()

    # list of channels
    pano_ = [frame[:,:, 0], frame[:, :, 1], frame[:, :, 2]]

    # set rotation
    p2p.set_rotation([0, 0, 0])

    # process image
    pers = np.array(p2p.process_image(pano_), copy=False)

    cv2.imshow("video", pers)

cap.release()
cv2.destroyAllWindows()
```

Convert by passing center coordinate point is also supported, however you would have to initialize the class using panorama image size and perspective FOV (both width and height):

```Python
import pano2perspective as pano

# Initialize the class with cropping size (width, height)
p2p = pano.Pano2Perspective(PANO_WIDTH, PANO_HEIGHT, 60.0, 45.0) # intialize using panorama image size and fovs
p2p.cuda(0)  # use cuda

cap = cv2.VideoCapture("path_to_video")
while(cap.isOpened()):
    _, frame = cap.read()

    # list of channels
    pano_ = [frame[:,:, 0], frame[:, :, 1], frame[:, :, 2]]

    # set rotation
    p2p.set_center_point(PANO_X, PANO_Y)

    # process image
    pers = np.array(p2p.process_image(pano_), copy=False)

    cv2.imshow("video", pers)

cap.release()
cv2.destroyAllWindows()
```

### Examples

Example usages are in `example.py`.

```
# test single image
python example.py --data="path/to/image"

# test video
python example.py --video --data="path/to/video"
```

## TODO:

- [x] Make build and install process easier
- [x] GPU selection
- [ ] Build when there are no GPUs available
- [ ] Test codes for C++
