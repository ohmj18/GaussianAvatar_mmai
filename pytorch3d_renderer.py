## PerspectiveCameras
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.common.datatypes import Device
from pytorch3d.transforms import Rotate, Transform3d, Translate

from .utils import convert_to_tensors_and_broadcast, TensorProperties


# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)

# An input which is a float per batch element
_BatchFloatType = Union[float, Sequence[float], torch.Tensor]

# one or two floats per batch element
_FocalLengthType = Union[
    float, Sequence[Tuple[float]], Sequence[Tuple[float, float]], torch.Tensor
]


class PerspectiveCameras(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.

    Parameters for this camera are specified in NDC if `in_ndc` is set to True.
    If parameters are specified in screen space, `in_ndc` must be set to False.
    """

    # For __getitem__
    _FIELDS = (
        "K",
        "R",
        "T",
        "focal_length",
        "principal_point",
        "_in_ndc",  # arg is in_ndc but attribute set as _in_ndc
        "image_size",
    )

    _SHARED_FIELDS = ("_in_ndc",)



    def __init__(
        self,
        focal_length: _FocalLengthType = 1.0,
        principal_point=((0.0, 0.0),),
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        K: Optional[torch.Tensor] = None,
        device: Device = "cpu",
        in_ndc: bool = True,
        image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> None:
        """

        Args:
            focal_length: Focal length of the camera in world units.
                A tensor of shape (N, 1) or (N, 2) for
                square and non-square pixels respectively.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (N, 2).
            in_ndc: True if camera parameters are specified in NDC.
                If camera parameters are in screen space, it must
                be set to False.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need focal_length, principal_point
            image_size: (height, width) of image size.
                A tensor of shape (N, 2) or a list/tuple. Required for screen cameras.
            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            _in_ndc=in_ndc,
            **kwargs,  # pyre-ignore
        )
        if image_size is not None:
            if (self.image_size < 1).any():  # pyre-ignore
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        # When focal length is provided as one value, expand to
        # create (N, 2) shape tensor
        if self.focal_length.ndim == 1:  # (N,)
            self.focal_length = self.focal_length[:, None]  # (N, 1)
        self.focal_length = self.focal_length.expand(-1, 2)  # (N, 2)





    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the projection matrix using the
        multi-view geometry convention.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in __init__.

        Returns:
            A `Transform3d` object with a batch of `N` projection transforms.

        .. code-block:: python

            fx = focal_length[:, 0]
            fy = focal_length[:, 1]
            px = principal_point[:, 0]
            py = principal_point[:, 1]

            K = [
                    [fx,   0,   px,   0],
                    [0,   fy,   py,   0],
                    [0,    0,    0,   1],
                    [0,    0,    1,   0],
            ]
        """
        K = kwargs.get("K", self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = "Expected K to have shape of (%r, 4, 4)"
                raise ValueError(msg % (self._N))
        else:
            K = _get_sfm_calibration_matrix(
                self._N,
                self.device,
                kwargs.get("focal_length", self.focal_length),
                kwargs.get("principal_point", self.principal_point),
                orthographic=False,
            )

        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device
        )
        return transform





    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        from_ndc: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            from_ndc: If `False` (default), assumes xy part of input is in
                NDC space if self.in_ndc(), otherwise in screen space. If
                `True`, assumes xy is in NDC space even if the camera
                is defined in screen space.
        """
        if world_coordinates:
            to_camera_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_camera_transform = self.get_projection_transform(**kwargs)
        if from_ndc:
            to_camera_transform = to_camera_transform.compose(
                self.get_ndc_camera_transform()
            )

        unprojection_transform = to_camera_transform.inverse()
        xy_inv_depth = torch.cat(
            (xy_depth[..., :2], 1.0 / xy_depth[..., 2:3]), dim=-1  # type: ignore
        )
        return unprojection_transform.transform_points(xy_inv_depth)





    def get_principal_point(self, **kwargs) -> torch.Tensor:
        """
        Return the camera's principal point

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.
        """
        proj_mat = self.get_projection_transform(**kwargs).get_matrix()
        return proj_mat[:, 2, :2]





    def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
        """
        Returns the transform from camera projection space (screen or NDC) to NDC space.
        If the camera is defined already in NDC space, the transform is identity.
        For cameras defined in screen space, we adjust the principal point computation
        which is defined in the image space (commonly) and scale the points to NDC space.

        This transform leaves the depth unchanged.

        Important: This transforms assumes PyTorch3D conventions for the input points,
        i.e. +X left, +Y up.
        """
        if self.in_ndc():
            ndc_transform = Transform3d(device=self.device, dtype=torch.float32)
        else:
            # when cameras are defined in screen/image space, the principal point is
            # provided in the (+X right, +Y down), aka image, coordinate system.
            # Since input points are defined in the PyTorch3D system (+X left, +Y up),
            # we need to adjust for the principal point transform.
            pr_point_fix = torch.zeros(
                (self._N, 4, 4), device=self.device, dtype=torch.float32
            )
            pr_point_fix[:, 0, 0] = 1.0
            pr_point_fix[:, 1, 1] = 1.0
            pr_point_fix[:, 2, 2] = 1.0
            pr_point_fix[:, 3, 3] = 1.0
            pr_point_fix[:, :2, 3] = -2.0 * self.get_principal_point(**kwargs)
            pr_point_fix_transform = Transform3d(
                matrix=pr_point_fix.transpose(1, 2).contiguous(), device=self.device
            )
            image_size = kwargs.get("image_size", self.get_image_size())
            screen_to_ndc_transform = get_screen_to_ndc_transform(
                self, with_xyflip=False, image_size=image_size
            )
            ndc_transform = pr_point_fix_transform.compose(screen_to_ndc_transform)

        return ndc_transform





    def is_perspective(self):
        return True





    def in_ndc(self):
        return self._in_ndc



## MeshRenderer
from typing import Tuple

import torch
import torch.nn as nn

from ...structures.meshes import Meshes

# A renderer class should be initialized with a
# function for rasterization and a function for shading.
# The rasterizer should:
#     - transform inputs from world -> screen space
#     - rasterize inputs
#     - return fragments
# The shader can take fragments as input along with any other properties of
# the scene and generate images.

# E.g. rasterize inputs and then shade
#
# fragments = self.rasterize(meshes)
# images = self.shader(fragments, meshes)
# return images


class MeshRenderer(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer (a MeshRasterizer or a MeshRasterizerOpenGL)
    and shader class which each have a forward function.
    """

    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader



    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self





    def forward(self, meshes_world: Meshes, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        For this set rasterizer.raster_settings.clip_barycentric_coords=True
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images



## MeshRasterizer
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch3d.renderer.cameras import try_get_projection_transform

from .rasterize_meshes import rasterize_meshes



@dataclass(frozen=True)
class Fragments:
    """
    A dataclass representing the outputs of a rasterizer. Can be detached from the
    computational graph in order to stop the gradients from flowing through the
    rasterizer.

    Members:
        pix_to_face:
            LongTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the indices of the nearest faces at each pixel, sorted in ascending
            z-order. Concretely ``pix_to_face[n, y, x, k] = f`` means that
            ``faces_verts[f]`` is the kth closest face (in the z-direction) to pixel
            (y, x). Pixels that are hit by fewer than faces_per_pixel are padded with
            -1.

        zbuf:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the NDC z-coordinates of the nearest faces at each pixel, sorted in
            ascending z-order. Concretely, if ``pix_to_face[n, y, x, k] = f`` then
            ``zbuf[n, y, x, k] = face_verts[f, 2]``. Pixels hit by fewer than
            faces_per_pixel are padded with -1.

        bary_coords:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel, 3)
            giving the barycentric coordinates in NDC units of the nearest faces at
            each pixel, sorted in ascending z-order. Concretely, if ``pix_to_face[n,
            y, x, k] = f`` then ``[w0, w1, w2] = barycentric[n, y, x, k]`` gives the
            barycentric coords for pixel (y, x) relative to the face defined by
            ``face_verts[f]``. Pixels hit by fewer than faces_per_pixel are padded
            with -1.

        dists:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the signed Euclidean distance (in NDC units) in the x/y plane of each
            point closest to the pixel. Concretely if ``pix_to_face[n, y, x, k] = f``
            then ``pix_dists[n, y, x, k]`` is the squared distance between the pixel
            (y, x) and the face given by vertices ``face_verts[f]``. Pixels hit with
            fewer than ``faces_per_pixel`` are padded with -1.
    """

    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: Optional[torch.Tensor]



    def detach(self) -> "Fragments":
        return Fragments(
            pix_to_face=self.pix_to_face,
            zbuf=self.zbuf.detach(),
            bary_coords=self.bary_coords.detach(),
            dists=self.dists.detach() if self.dists is not None else self.dists,
        )


@dataclass
class RasterizationSettings:
    """
    Class to store the mesh rasterization params with defaults

    Members:
        image_size: Either common height and width or (height, width), in pixels.
        blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
        faces_per_pixel: (int) Number of faces to keep track of per pixel.
            We return the nearest faces_per_pixel faces along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts
            to set it heuristically based on the shape of the input. This should
            not affect the output, but can affect the speed of the forward pass.
        max_faces_opengl: Max number of faces in any mesh we will rasterize. Used only by
            MeshRasterizerOpenGL to pre-allocate OpenGL memory.
        max_faces_per_bin: Only applicable when using coarse-to-fine
            rasterization (bin_size != 0); this is the maximum number of faces
            allowed within each bin. This should not affect the output values,
            but can affect the memory usage in the forward pass.
            Setting max_faces_per_bin=None attempts to set with a heuristic.
        perspective_correct: Whether to apply perspective correction when
            computing barycentric coordinates for pixels.
            None (default) means make correction if the camera uses perspective.
        clip_barycentric_coords: Whether, after any perspective correction
            is applied but before the depth is calculated (e.g. for
            z clipping), to "correct" a location outside the face (i.e. with
            a negative barycentric coordinate) to a position on the edge of the
            face. None (default) means clip if blur_radius > 0, which is a condition
            under which such outside-face-points are likely.
        cull_backfaces: Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.
        z_clip_value: if not None, then triangles will be clipped (and possibly
            subdivided into smaller triangles) such that z >= z_clip_value.
            This avoids camera projections that go to infinity as z->0.
            Default is None as clipping affects rasterization speed and
            should only be turned on if explicitly needed.
            See clip.py for all the extra computation that is required.
        cull_to_frustum: Whether to cull triangles outside the view frustum.
            Culling involves removing all faces which fall outside view frustum.
            Default is False for performance as often not needed.
    """

    image_size: Union[int, Tuple[int, int]] = 256
    blur_radius: float = 0.0
    faces_per_pixel: int = 1
    bin_size: Optional[int] = None
    max_faces_opengl: int = 10_000_000
    max_faces_per_bin: Optional[int] = None
    perspective_correct: Optional[bool] = None
    clip_barycentric_coords: Optional[bool] = None
    cull_backfaces: bool = False
    z_clip_value: Optional[float] = None
    cull_to_frustum: bool = False


class MeshRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogeneous
    Meshes.
    """



    def __init__(self, cameras=None, raster_settings=None) -> None:
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings





    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self





    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.

        Returns:
            meshes_proj: a Meshes object with the vertex positions projected
            in NDC space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        eps = kwargs.get("eps", None)
        verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            verts_world, eps=eps
        )
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = try_get_projection_transform(cameras, kwargs)
        if projection_transform is not None:
            projection_transform = projection_transform.compose(to_ndc_transform)
            verts_ndc = projection_transform.transform_points(verts_view, eps=eps)
        else:
            # Call transform_points instead of explicitly composing transforms to handle
            # the case, where camera class does not have a projection matrix form.
            verts_proj = cameras.transform_points(verts_world, eps=eps)
            verts_ndc = to_ndc_transform.transform_points(verts_proj, eps=eps)

        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)
        return meshes_ndc




    def forward(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_proj = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct and z_clip_value from the camera
        cameras = kwargs.get("cameras", self.cameras)
        if raster_settings.perspective_correct is not None:
            perspective_correct = raster_settings.perspective_correct
        else:
            perspective_correct = cameras.is_perspective()
        if raster_settings.z_clip_value is not None:
            z_clip = raster_settings.z_clip_value
        else:
            znear = cameras.get_znear()
            if isinstance(znear, torch.Tensor):
                znear = znear.min().item()
            z_clip = None if not perspective_correct or znear is None else znear / 2

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=raster_settings.cull_to_frustum,
        )

        return Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
        )




## BlendParams
from typing import NamedTuple, Sequence, Union

import torch
from pytorch3d import _C
from pytorch3d.common.datatypes import Device

# Example functions for blending the top K colors per pixel using the outputs
# from rasterization.
# NOTE: All blending function should return an RGBA image per batch element



class BlendParams(NamedTuple):
    """
    Data class to store blending params with defaults

    Members:
        sigma (float): For SoftmaxPhong, controls the width of the sigmoid
            function used to calculate the 2D distance based probability. Determines
            the sharpness of the edges of the shape. Higher => faces have less defined
            edges. For SplatterPhong, this is the standard deviation of the Gaussian
            kernel. Higher => splats have a stronger effect and the rendered image is
            more blurry.
        gamma (float): Controls the scaling of the exponential function used
            to set the opacity of the color.
            Higher => faces are more transparent.
        background_color: RGB values for the background color as a tuple or
            as a tensor of three floats.
    """

    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (1.0, 1.0, 1.0)





## SoftSilhouetteShader
import warnings
from typing import Optional

import torch
import torch.nn as nn

from ...common.datatypes import Device
from ...structures.meshes import Meshes
from ..blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from ..lighting import PointLights
from ..materials import Materials
from ..splatter_blend import SplatterBlender
from ..utils import TensorProperties
from .rasterizer import Fragments
from .shading import (
    _phong_shading_with_pixels,
    flat_shading,
    gouraud_shading,
    phong_shading,
)


# A Shader should take as input fragments from the output of rasterization
# along with scene params and output images. A shader could perform operations
# such as:
#     - interpolate vertex attributes for all the fragments
#     - sample colors from a texture map
#     - apply per pixel lighting
#     - blend colors across top K faces per pixel.


class SoftSilhouetteShader(nn.Module):
    """
    Calculate the silhouette by blending the top K faces for each pixel based
    on the 2d euclidean distance of the center of the pixel to the mesh face.

    Use this shader for generating silhouettes similar to SoftRasterizer [0].

    .. note::

        To be consistent with SoftRasterizer, initialize the
        RasterizationSettings for the rasterizer with
        `blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma`

    [0] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
        3D Reasoning', ICCV 2019
    """

    def __init__(self, blend_params: Optional[BlendParams] = None) -> None:
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()



    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        """
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        colors = torch.ones_like(fragments.bary_coords)
        blend_params = kwargs.get("blend_params", self.blend_params)
        images = sigmoid_alpha_blend(colors, fragments, blend_params)
        return images
