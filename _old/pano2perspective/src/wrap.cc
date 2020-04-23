#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include "pano2perspective.hh"

namespace py = pybind11;

PYBIND11_MODULE(pano2perspective, m)
{
    m.doc() = "Pano2Perspective: a library for converting panorama to perspective image";

    m.def("process_single_image", &process_single_image, "A function to process 1 panorama image and output a perspective based on rotation");

    py::class_<Pano2Perspective>(m, "Pano2Perspective")
        .def(py::init<int, int, float>(), py::arg("w")=480, py::arg("h")=360, py::arg("fov")=90.0)
        .def(py::init<int, int, float, float>())
        .def("process_image", &Pano2Perspective::process_image)
        .def("get_intrinsics", &Pano2Perspective::get_intrinsics)
        .def("set_rotation", &Pano2Perspective::set_rotation)
        .def("set_center_point", &Pano2Perspective::set_center_point)
        .def("cuda", &Pano2Perspective::cuda);

    // Buffer protocol for return value
    //
    // Turn `Image` into np.array by:
    // np.array(<this library>.get_image(src, rot, size), copy=False)
    py::class_<cv::Mat>(m, "Image", py::buffer_protocol())
        .def_buffer([](cv::Mat& im) -> py::buffer_info{
            return py::buffer_info(
                // Pointer to buffer
                im.data,
                // Size of one scalar
                sizeof(unsigned char),
                // Python struct-style format descriptor
                py::format_descriptor<unsigned char>::format(),
                // Number of dimensions
                3,
                // Buffer dimensions
                { im.rows, im.cols, im.channels() },
                // Strides (in bytes) for each index
                {
                    sizeof(unsigned char) * im.channels() * im.cols,
                    sizeof(unsigned char) * im.channels(),
                    sizeof(unsigned char)
                }
            );
        });
}
