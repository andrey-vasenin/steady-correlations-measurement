//
// Created by andrei on 4/13/21.
//
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "measurement.cuh"

using namespace pybind11::literals;

namespace py = pybind11;

PYBIND11_MODULE(mollowgpu, m) {
    py::class_<Measurement>(m, "MollowGpu")
        .def(py::init<std::uintptr_t, unsigned long long, int, float>())
        .def("set_calibration", &Measurement::setCalibration)
        .def("measure", &Measurement::measure)
        .def("reset", &Measurement::reset)
        .def("free", &Measurement::free);
}
