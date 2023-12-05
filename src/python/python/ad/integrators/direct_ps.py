from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from .common import PSIntegrator, mis_weight

class DirectProjectiveIntegrator(PSIntegrator):
    r"""
    .. _integrator-direct_proj:
    - No radiative backpropagation
    - Detached sampling to compute continuous deriv
    """

    def __init__(self, props):
        super().__init__(props)

        # Override the max depth to 2 since this is a direct illumination
        # integrator
        self.max_depth = 2
        self.rr_depth = 2
        # Direct illumination integrators don't need radiative backpropagation
        self.radiative_backprop = False

        # Specify the seed ray generation strategy
        self.project_seed = props.get('project_seed', 'both')
        if self.project_seed not in ['both', 'bsdf', 'emitter']:
            raise Exception(f"Project seed must be one of 'both', 'bsdf', "
                            f"'emitter', got '{self.project_seed}'")


    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL: Optional[mi.Spectrum],
               state_in: Any,
               active: mi.Bool,
               project: bool = False,
               si_shade: Optional[mi.SurfaceInteraction3f] = None,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, Any]:
        """
        See ``PSIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """
        del depth, δL, state_in, kwargs  # Unused

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Should we use ``si_shade`` as the first interaction and ignore the ray?
        ignore_ray = si_shade is not None

        # Standard BSDF evaluation context
        bsdf_ctx = mi.BSDFContext()

        L = mi.Spectrum(0)

        # ---------------------- Direct emission ----------------------

        # Use `si_shade` as the first interaction or trace a ray to the first
        # interaction
        if ignore_ray:
            si = si_shade
        else:
            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All,
                             coherent=True, active=active)

        # Hide the environment emitter if necessary
        if not self.hide_emitters:
            with dr.resume_grad(when=not primal):
                L += si.emitter(scene).eval(si, active)

        active_next = active & si.is_valid() & (self.max_depth > 1)

        # Get the BSDF
        bsdf = si.bsdf(ray)

        # ---------------------- Emitter sampling ----------------------

        # Is emitter sampling possible on the current vertex?
        active_em_ = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        # If so, pick an emitter and sample a detached emitter direction
        ds_em, emitter_val = scene.sample_emitter_direction(
            si, sampler.next_2d(active_em_), test_visibility=True, active=active_em_)
        active_em = active_em_ & dr.neq(ds_em.pdf, 0.0)

        with dr.resume_grad(when=not primal):
            # Evaluate the BSDF (foreshortening term included)
            wo = si.to_local(ds_em.d)
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)

            # Re-compute some values with AD attached only in differentiable
            # phase
            if not primal:
                # Re-compute attached `emitter_val` to enable emitter optimization
                # TODO: zz: do we really need this?
                ds_em.d = dr.normalize(ds_em.p - si.p)
                spec_em = scene.eval_emitter_direction(si, ds_em, active_em)
                emitter_val = spec_em / ds_em.pdf
                dr.disable_grad(ds_em.d)

            # Compute the detached MIS weight for the emitter sample
            mis_em = dr.select(ds_em.delta, 1.0, mis_weight(ds_em.pdf, bsdf_pdf))

            L[active_em] += bsdf_val * emitter_val * mis_em

        # ---------------------- BSDF sampling ----------------------

        # Perform detached BSDF sampling
        sample_bsdf, weight_bsdf = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active_next),
                                               sampler.next_2d(active_next), active_next)
        active_bsdf = active_next & dr.any(dr.neq(weight_bsdf, 0.0))
        delta_bsdf = mi.has_flag(sample_bsdf.sampled_type, mi.BSDFFlags.Delta)

        # Construct the BSDF sampled ray
        ray_bsdf = si.spawn_ray(si.to_world(sample_bsdf.wo))

        with dr.resume_grad(when=not primal):
            # Re-compute `weight_bsdf` with AD attached only in differentiable
            # phase
            if not primal:
                wo = si.to_local(ray_bsdf.d)
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_bsdf)
                weight_bsdf = bsdf_val / dr.detach(bsdf_pdf)

                # ``ray_bsdf`` is left detached (both origin and direction)

            # Trace the BSDF sampled ray
            si_bsdf = scene.ray_intersect(
                ray_bsdf, ray_flags=mi.RayFlags.All, coherent=False, active=active_bsdf)

            # Evaluate the emitter
            L_bsdf = si_bsdf.emitter(scene, active_bsdf).eval(si_bsdf, active_bsdf)

            # Compute the detached MIS weight for the BSDF sample
            ds_bsdf = mi.DirectionSample3f(scene, si=si_bsdf, ref=si)
            pdf_emitter = scene.pdf_emitter_direction(
                ref=si, ds=ds_bsdf, active=active_bsdf & ~delta_bsdf)
            mis_bsdf = dr.select(delta_bsdf, 1.0, mis_weight(sample_bsdf.pdf, pdf_emitter))

            L[active_bsdf] += L_bsdf * weight_bsdf * mis_bsdf

        # ---------------------- Seed rays for projection ----------------------

        guide_seed = []
        if project:
            if self.project_seed == "bsdf":
                # A BSDF sample can only be a valid seed ray if it intersects a
                # shape
                active_guide = active_bsdf & si_bsdf.is_valid()

                guide_seed = [dr.detach(ray_bsdf), mi.Bool(active_guide)]
            elif self.project_seed == "emitter":
                # Return emitter sampling rays including the ones that fail
                # `test_visibility`
                ray_em = si.spawn_ray_to(ds_em.p)
                ray_em.maxt = dr.largest(mi.Float)

                # Directions towards the interior have no contribution for
                # direct integrators
                active_guide = active_em_ & (dr.dot(si.n, ray_em.d) > 0)

                guide_seed = [dr.detach(ray_em), mi.Bool(active_guide)]
            elif self.project_seed == "both":
                # By default we use the emitter sample as the seed ray
                ray_seed = si.spawn_ray_to(ds_em.p)
                ray_seed.maxt = dr.largest(mi.Float)
                active_guide = active_em_ & (dr.dot(si.n, ray_seed.d) > 0)

                # Flip a coin only when both samples are valid
                mask_replace = (active_bsdf & si_bsdf.is_valid()) & \
                               ((sampler.next_1d() > 0.5) | ~active_guide)
                ray_seed[mask_replace] = ray_bsdf

                guide_seed = [dr.detach(ray_seed), active_guide | mask_replace]

        return L, active, guide_seed if project else None


    def sample_radiance_difference(self, scene, ss, curr_depth, sampler, active):
        if curr_depth == 1:

            # ----------- Estimate the radiance of the background -----------

            ray_bg = ss.spawn_ray()
            si_bg = scene.ray_intersect(ray_bg, active=active)
            radiance_bg = si_bg.emitter(scene).eval(si_bg, active)

            # ----------- Estimate the radiance of the foreground -----------

            # For direct illumination integrators, only an area emitter can
            # contribute here. It is possible to call ``sample()`` to estimate
            # this contribution. But to avoid the overhead we simply query the
            # emitter here to obtain the radiance.
            si_fg = dr.zeros(mi.SurfaceInteraction3f)

            # We know the incident direction is valid since this is the
            # foreground interaction. Overwrite the incident direction to avoid
            # potential issues introduced by smooth normals.
            si_fg.wi = mi.Vector3f(0, 0, 1)
            radiance_fg = ss.shape.emitter().eval(si_fg, active)
        elif curr_depth == 0:

            # ----------- Estimate the radiance of the background -----------
            ray_bg = ss.spawn_ray()
            radiance_bg, _, _ = self.sample(
                dr.ADMode.Primal, scene, sampler, ray_bg, curr_depth, None, None, active, False, None)

            # ----------- Estimate the radiance of the foreground -----------
            # Create a preliminary intersection point
            pi_fg = dr.zeros(mi.PreliminaryIntersection3f)
            pi_fg.t = 1
            pi_fg.prim_index = ss.prim_index
            pi_fg.prim_uv = ss.uv
            pi_fg.shape = ss.shape

            # Create a dummy ray that we never perform ray-intersection with to
            # compute other fields in ``si``
            dummy_ray = mi.Ray3f(ss.p - ss.d, ss.d)

            # The ray origin is wrong, but this is fine if we only need the primal
            # radiance
            si_fg = pi_fg.compute_surface_interaction(
                dummy_ray, mi.RayFlags.All, active)

            # If smooth normal is used, it is possible that the computed shading
            # normal near visibility silhouette points to the wrong side of the
            # surface. We fix this by clamping the shading frame normal to the
            # visible side.
            if True:
                alpha = dr.dot(si_fg.sh_frame.n, ss.d)
                eps = dr.epsilon(alpha) * 100
                wrong_side = active & (alpha > -eps)

                # NOTE: In the following case, (1) a single sided BSDF is used,
                # (2) the silhouette sample is on an open boundary like an open
                # edge, and (3) we actually hit the back side of the surface,
                # the expected radiance is zero because no BSDF is defiend on
                # that side. But this shading frame correction will mistakenly
                # produce a non-zero radiance. Please use two-sided BSDFs if
                # this is a concern.
                # The following does not fix the issue since `si_fg.n` might
                # be the geometric normal of the backfacing primitive.
                #    bsdf = si_fg.bsdf()
                #    wrong_side &= mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) | \
                #                  (dr.dot(ss.d, si_fg.n) < 0.01)

                # Remove the component of the shading frame normal that points to
                # the wrong side
                new_sh_normal = dr.normalize(
                    si_fg.sh_frame.n - (alpha + eps) * ss.d)
                # Start ``si_fg`` surgery
                si_fg.sh_frame[wrong_side] = mi.Frame3f(new_sh_normal)
                si_fg.wi[wrong_side] = si_fg.to_local(-ss.d)

                # print(f"wrong side = {dr.count(active & (si_fg.wi.z < 0))}")  # Should give 0

            # Estimate the radiance starting from the surface interaction
            radiance_fg, _, _ = self.sample(
                dr.ADMode.Primal, scene, sampler, ray_bg, curr_depth, None, None, active, False, si_fg)

        else:
            raise Exception(f"Unexpected depth {curr_depth} in direct projective integrator")

        # Compute the radiance difference
        radiance_diff = radiance_fg - radiance_bg
        active_diff = active & (dr.max(dr.abs(radiance_diff)) > 0)

        return radiance_diff, active_diff


    def sample_importance(self, scene, sensor, ss, max_depth, sampler, preprocess, active):
        del max_depth  # Unused

        # Trace a ray to the camera ray intersection
        ss_importance = mi.SilhouetteSample3f(ss)
        ss_importance.d = -ss_importance.d 
        ray_boundary = ss_importance.spawn_ray()
        if preprocess:
            si_boundary = scene.ray_intersect(ray_boundary, active=active)
        else:
            with dr.resume_grad():
                si_boundary = scene.ray_intersect(
                    ray_boundary,
                    ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                    coherent=False,
                    active=active)
        active_i = active & si_boundary.is_valid()

        # Connect `si_boundary` to the sensor
        it = dr.zeros(mi.Interaction3f)
        it.p = si_boundary.p
        sensor_ds, sensor_weight = sensor.sample_direction(it, sampler.next_2d(active_i), active_i)
        active_i &= dr.neq(sensor_ds.pdf, 0)

        # Visibility to sensor
        cam_test_ray = si_boundary.spawn_ray_to(sensor_ds.p)
        active_i &= ~scene.ray_test(cam_test_ray, active_i)

        # Recompute the correct motion of the first interaction point (camera
        # ray intersection in the direct illumination integrator)
        if not preprocess:
            d = dr.normalize(sensor_ds.p - si_boundary.p)
            O = si_boundary.p - d
            with dr.resume_grad():
                t = dr.dot(si_boundary.p - O, si_boundary.n) / dr.dot(d, si_boundary.n)
                si_boundary.p = dr.replace_grad(si_boundary.p, O + t * d)

        # Evaluate the BSDF
        bsdf_ctx = mi.BSDFContext(mi.TransportMode.Importance)
        wo_local = si_boundary.to_local(sensor_ds.d)
        bsdf_val = si_boundary.bsdf().eval(
            bsdf_ctx, si_boundary, wo_local, active_i)
        active_i &= dr.neq(dr.max(bsdf_val), 0)

        importance = bsdf_val * sensor_weight
        return importance, sensor_ds.uv, mi.UInt32(2), si_boundary.p, active_i

mi.register_integrator("direct_projective", lambda props: DirectProjectiveIntegrator(props))
