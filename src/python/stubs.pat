# This file contains declarations to patch Mitsuba's stub files. nanobind's
# stubgen automatically applies them during the build process
#
# The syntax of this file is described here:
#
# https://nanobind.readthedocs.io/en/latest/typing.html#pattern-files
#
# The design of type signatures and the use of generics and type variables is
# explained in the Dr.Jit documentation section entitled "Type Signatures".
#
# Whenever possible, it's preferable to specify signatures to C++ bindings
# using the nb::sig() override. The rules below are used in cases where that is
# not possible, or when the typing-specific overloads of a funciton deviate
# significantly from the overload chain implemented using nanobind.


# --------------------------- Core -----------------------------

mitsuba_ext.__prefix__:
    import mitsuba.scalar_rgb

MI_AUTHORS:
MI_ENABLE_CUDA:
MI_ENABLE_EMBREE:
MI_FILTER_RESOLUTION:
MI_VERSION:
MI_VERSION_MAJOR:
MI_VERSION_MINOR:
MI_VERSION_PATCH:
MI_YEAR:
DEBUG:

cast_object:
casters:

Appender.log_progress:
    def log_progress(self, progress: float, name: str, formatted: str, eta: str, ptr: object = None) -> None:
        \doc

Object.traverse:
    def traverse(self, cb: TraversalCallback) -> None:
        \doc

Field.__eq__:
Field.__ne__:

Bitmap.__init__:
    def __init__(self, pixel_format: Bitmap.PixelFormat, component_format: Struct.Type, size: mitsuba.scalar_rgb.Vector2u, channel_count: int = 0, channel_names: Sequence[str] = []) -> None:
        \doc

Bitmap.__eq__:
Bitmap.__ne__:

Bitmap.metadata:
    def metadata(self) -> mitsuba.scalar_rgb.Properties:
        \doc

Bitmap.resample:
    @overload
    def resample(self, target: Bitmap, rfilter: mitsuba.scalar_rgb.ReconstructionFilter | None = None, bc: tuple[FilterBoundaryCondition, FilterBoundaryCondition] = ..., clamp: tuple[float, float] = (float('-inf'), float('inf')), temp: Bitmap | None = None) -> None:
        \doc

    @overload
    def resample(self, res: mitsuba.scalar_rgb.Vector2u, rfilter: mitsuba.scalar_rgb.ReconstructionFilter | None = None, bc: tuple[FilterBoundaryCondition, FilterBoundaryCondition] = ..., clamp: tuple[float, float] = (float('-inf'), float('inf'))) -> Bitmap:
        \doc

Bitmap.accumulate:
    @overload
    def accumulate(self, bitmap: Bitmap, source_offset: mitsuba.scalar_rgb.Point2i, target_offset: mitsuba.scalar_rgb.Point2i, size: mitsuba.scalar_rgb.Vector2i) -> None:
        \doc

    @overload
    def accumulate(self, bitmap: Bitmap, target_offset: mitsuba.scalar_rgb.Point2i) -> None:
        \doc

Bitmap.size:
    def size(self) -> mitsuba.scalar_rgb.Vector2i:
        \doc

PluginManager.create_object:
    def create_object(self, arg: mitsuba.scalar_rgb.Properties, /) -> object:
        \doc

Resampler.__init__:
    def __init__(self, rfilter: mitsuba.scalar_rgb.ReconstructionFilter, source_res: int, target_res: int) -> None:
        \doc

Spiral.__init__:
    def __init__(self, size: mitsuba.scalar_rgb.Vector2u, offset: mitsuba.scalar_rgb.Vector2u, block_size: int = 32, passes: int = 1) -> None:
        \doc

Spiral.next_block:
    def next_block(self) -> tuple[mitsuba.scalar_rgb.Vector2i, mitsuba.scalar_rgb.Vector2u, int]:
        \doc

Struct.append:
Struct.__eq__:
Struct.__ne__:

parse_fov:
    def parse_fov(props: mitsuba.scalar_rgb.Properties, aspect: float) -> float:
        \doc


# ------------------------- Variant ----------------------------

.*\.DRJIT_STRUCT:

Scalar.*:

UnpolarizedSpectrum:

is_monochromatic:
is_polarized:
is_rgb:
is_spectral:
sggx_sample:
sggx_pdf:
sggx_projected_area:


Vector.*:
Array.*:
Point.*:
Color.*:

_.*Cp:

.*Ptr.Domain:
.*Ptr.__getitem__:
.*Ptr.__setitem__:
.*Ptr.__delitem__:

Transform.+d:

Ray.+d:

