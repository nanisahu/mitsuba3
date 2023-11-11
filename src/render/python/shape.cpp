#include <mitsuba/render/shape.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(DiscontinuityFlags) {
    auto e = py::enum_<DiscontinuityFlags>(m, "DiscontinuityFlags", D(DiscontinuityFlags))
        .def_value(DiscontinuityFlags, Empty)
        .def_value(DiscontinuityFlags, PerimeterType)
        .def_value(DiscontinuityFlags, InteriorType)
        .def_value(DiscontinuityFlags, DirectionLune)
        .def_value(DiscontinuityFlags, DirectionSphere)
        .def_value(DiscontinuityFlags, AllTypes);

        MI_PY_DECLARE_ENUM_OPERATORS(DiscontinuityFlags, e)
}
