from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr
import numpy as np
import time

# TODO, we have two options for the 1d distr:
# (1) mitsuba's DiscreteDistribution
# (2) alias sampling (not implemented yet)

# Global variable to keep track of previous timestamp (used by `log_runtime`)
prev_time = 0


def get_kernels(only_jit=True):
    """
    Helper to fetch and filter kernels from the History and only keep relevant
    keys.
    """
    kernels = dr.kernel_history()
    filtered_kernels = []
    for kernel in kernels:
        if (not only_jit) or kernel["type"] == dr.KernelType.JIT:
            filtered_kernels.append(kernel)

    kernels = []
    keys = ['execution_time', 'backend_time', 'codegen_time', 'operation_count']
    for kernel in filtered_kernels:
        kernel = {key: kernel[key] if key in kernel else 0 for key in keys}
        kernels.append(kernel)

    return kernels


def log_runtime(msg, active, initial=False, print_kernel=False):
    """
    Helper to log runtimes and print kernels
    """
    if active:
        global prev_time
        dr.eval()
        dr.sync_thread()
        if initial:
            prev_time = time.time() * 1000

        current_time = time.time() * 1000
        runtime = current_time - prev_time
        prev_time = current_time

        if print_kernel:
            kernels = get_kernels()
            for kernel in kernels:
                print(f"{kernel}")

        if not initial:
            print(f"{msg}{runtime:.5f} [ms]")
        else:
            print(f"{msg}")

############################################################################################################

class BaseGuidingDistr():
    # Return a sample in U^3 from the stored guiding distribution and its reciprocal density.
    def sample(self, sampler):
        raise NotImplementedError
    def vis_query(self, reso=None):
        raise NotImplementedError


class UniformDistr(BaseGuidingDistr):
    # No guiding.
    def sample(self, sampler):
        return mi.Vector3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d()), mi.Float(1)


class GridDistr(BaseGuidingDistr):
    # Regular grid guiding.
    def __init__(self, resolution, clamp_mass_thres, sqrt_scale_mass=False, _DEBUG_LOG=False) -> None:
        self.prod_reso = []
        self.unit_delta = []
        self.num_cells = 1
        for x in reversed(resolution):
            self.num_cells *= x
            self.prod_reso.insert(0, self.num_cells)  ## FIXME: don't need this
            self.unit_delta.insert(0, dr.rcp(x))
        self.resolution = resolution
        self.clamp_mass_thres = clamp_mass_thres
        self.sqrt_scale_mass = sqrt_scale_mass
        self._DEBUG_LOG = _DEBUG_LOG

    def get_cell_array(self, index_array_):
        # Returns the integer cell array corresponding to the index array.
        # With `index_array`=dr.arange(mi.UInt32, self.num_cells), the output array is
        # [[0, 0, 0], [0, 0, 1], ..., [Nx-1, Ny-1, Nz-1]].
        index_array = mi.UInt32(index_array_)
        cells = dr.zeros(mi.Vector3u, dr.width(index_array))
        cells.x = index_array // self.prod_reso[1]
        index_array -= self.prod_reso[1] * cells.x
        cells.y = index_array // self.prod_reso[2]
        index_array -= cells.y * self.prod_reso[2]
        cells.z = index_array
        return cells

    def set_mass(self, mass):
        # Optional tradeoff between variance and (negligible) bias: Regions
        # whose contribution is many magnitudes smaller than others are very
        # rarely sampled in the rendering phase. But once sampled, they
        # contribute a lot to the variance since the reciprocal probability is
        # huge (>1e8). We can clamp the mass of these regions to suppress these
        # fireflies. This can be turned off if we have a large sampling budget.
        if self.clamp_mass_thres > 0:
            mass_ttl = dr.sum(mass)
            thres = self.clamp_mass_thres * mass_ttl
            clamp_mask = mass < thres
            mass[clamp_mask] = 0
            dr.schedule(mass, mass_ttl)
            if self._DEBUG_LOG: print(f"GridDistr: mass clamped with threshold {thres} (ttl mass = {mass_ttl}) [{dr.width(mass)-dr.count(clamp_mask)}/{dr.width(mass)}={(dr.width(mass)-dr.count(clamp_mask))/dr.width(mass)*100.:.3f}%]")

        if self.sqrt_scale_mass:
            mass = dr.safe_sqrt(mass)

        self.pmf = mi.DiscreteDistribution(mass)
        dr.eval(self.pmf)
        dr.sync_thread()

    def sample(self, sampler):
        sample1 = sampler.next_1d()
        sample2 = sampler.next_1d()
        sample_z = sampler.next_1d()

        idx, prob = self.pmf.sample_pmf(sampler.next_1d())
        rcp_prob = dr.rcp(prob)

        sample_new = self.get_cell_array(idx)
        sample_new += mi.Vector3f(sample1, sample2, sample_z)
        sample_new *= self.unit_delta

        # NOTE: The rcp pdf of randomly selecting a point in one cell is (1 / num_cells)
        rcp_pdf = dr.rcp(self.num_cells) * rcp_prob

        # WHY? needs to be fixed # TODO
        rcp_pdf[dr.eq(prob, 0)] = 0

        return sample_new, rcp_pdf

    # Generate one random sample in each cell.
    # Return the generated samples in [0, 1]^3.
    def random_cell_sample(self, sampler):
        idx = dr.arange(mi.UInt32, self.num_cells)
        sample_res = self.get_cell_array(idx)
        sample_res += mi.Vector3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())
        return sample_res * self.unit_delta

    # Return the index of the cell that contains the sample.
    def sample_to_cell_idx(self, sample, active=True):
        idx_3 = mi.Point3u(sample * self.resolution)
        index = idx_3.x * mi.UInt32(self.prod_reso[1]) + idx_3.y * mi.UInt32(self.prod_reso[2]) + idx_3.z
        return dr.select(active, index, 0)

    # Visualization purpose.
    def vis_query(self):
        pts = (self.get_cell_array(dr.arange(mi.UInt32, self.num_cells)) + 0.5) * self.unit_delta
        mass_ = self.pmf.pmf() * (self.pmf.normalization() * 1e6)
        return (pts.numpy(), mass_.numpy())

    def __repr__(self) -> str:
        return f"GridDistr [resolution={self.resolution}, num_cells={self.num_cells}]"


class OcSpaceDistr(BaseGuidingDistr):
    def __init__(self, max_depth, max_leaf_count, extra_spc, eval_indirect_integrand_handle,
                 clamp_input_mass_thres, clamp_mass_thres, prepartition_x_slices,
                 scatter_inc=False, sqrt_scale_mass=False, _DEBUG_LOG=False) -> None:
        self.max_depth = max_depth
        self.max_leaf_count = max_leaf_count
        self.extra_spc = extra_spc
        self.eval_indirect_integrand_handle = eval_indirect_integrand_handle
        self.prepartition_x_slices = prepartition_x_slices
        self.clamp_input_mass_thres = clamp_input_mass_thres
        self.clamp_mass_thres = clamp_mass_thres
        self.scatter_inc = scatter_inc
        self.sqrt_scale_mass = sqrt_scale_mass
        self._DEBUG_LOG = _DEBUG_LOG

    @staticmethod
    def aabbs(buffer: mi.Float, node_idx: mi.UInt32):
        idx = node_idx * 2
        aabb_min = dr.gather(mi.Point3f, buffer, idx)
        aabb_max = dr.gather(mi.Point3f, buffer, idx + 1)

        return (aabb_min, aabb_max)


    @staticmethod
    def split_offset(x_offset, y_offset, z_offset):
        return x_offset * 1 + y_offset * 2 + z_offset * 4


    @staticmethod
    def split(
            buffer: mi.Float,
            aabb_min: mi.Point3f,
            aabb_max: mi.Point3f,
            aabb_middle: mi.Point3f,
            node_idx: mi.UInt32
        ):

        def write_aabb(aabb_min: mi.Point3f, aabb_max: mi.Point3f, offset: int):
            buffer_offset = dr.opaque(mi.UInt32, dr.width(node_idx) * 6 * offset)

            # TODO: write as two `dr.scatter()` operations of `mi.Point3f` types
            dr.scatter(buffer, aabb_min.x, node_idx * 6 + buffer_offset + 0)
            dr.scatter(buffer, aabb_min.y, node_idx * 6 + buffer_offset + 1)
            dr.scatter(buffer, aabb_min.z, node_idx * 6 + buffer_offset + 2)

            dr.scatter(buffer, aabb_max.x, node_idx * 6 + buffer_offset + 3)
            dr.scatter(buffer, aabb_max.y, node_idx * 6 + buffer_offset + 4)
            dr.scatter(buffer, aabb_max.z, node_idx * 6 + buffer_offset + 5)

        x = [aabb_min.x, aabb_middle.x, aabb_max.x]
        y = [aabb_min.y, aabb_middle.y, aabb_max.y]
        z = [aabb_min.z, aabb_middle.z, aabb_max.z]

        part_min = dr.zeros(mi.Vector3f, dr.width(node_idx))
        part_max = dr.zeros(mi.Vector3f, dr.width(node_idx))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    part_min.x = x[i+0]; part_min.y = y[j+0]; part_min.z = z[k+0]
                    part_max.x = x[i+1]; part_max.y = y[j+1]; part_max.z = z[k+1]
                    offset = OcSpaceDistr.split_offset(i, j, k)
                    write_aabb(part_min, part_max, offset)


    def construct_octree(self, points, log=False):
        """
        Main Octree construction code

        6 phases:
            * Split non-leave nodes into 8
            * Figure out indices of points/samples after the split
            * Determine which of the split nodes are leaves
            * Write leave nodes to special buffer
            * Nodes that are still not leaves must be copied back to a buffer
              for the next iterations
            * Figure out indices of points/samples that are still not in leaves
        """
        log_runtime("Octree construction initialization:", log, initial=True)
        # Holds the list of AABBs that needs to be split
        # This is a flat buffer with the following structure:
        # [
        #  aabb_min_x[0], aabb_min_y[0], aabb_min_z[0],
        #  aabb_max_x[0], aabb_max_y[0], aabb_max_z[0],
        #  aabb_min_x[1], aabb_min_y[1], aabb_min_z[1],
        #  aabb_max_x[1], aabb_max_y[1], aabb_max_z[1],
        #  ...
        #  aabb_min_x[N - 1], aabb_min_y[N - 1], aabb_min_z[N - 1],
        #  aabb_max_x[N - 1], aabb_max_y[N - 1], aabb_max_z[N - 1],
        # ]
        aabb_buffer = dr.zeros(mi.Float, self.max_leaf_count * 6)
        aabb_buffer_next = dr.zeros(mi.Float, self.max_leaf_count * 6)
        leaves = dr.zeros(mi.Float, self.max_leaf_count * 6)
        leaves_count = mi.UInt32(0)
        dr.make_opaque(leaves_count)

        INVALID = mi.UInt32(0xffffffff)
        points_idx = dr.full(mi.UInt32, np.uint32(INVALID), dr.width(points))

        #########################
        # Initialize root nodes #
        #########################
        active_node_count = self.prepartition_x_slices

        idx = dr.arange(mi.UInt32, active_node_count)
        zero_vec = dr.zeros(mi.Vector3f, active_node_count)
        one_vec = dr.ones(mi.Vector3f, active_node_count)
        dr.scatter(aabb_buffer, zero_vec, idx * 2 + 0)
        dr.scatter(aabb_buffer, one_vec,  idx * 2 + 1)

        x = dr.linspace(mi.Float, 0, 1, active_node_count + 1)
        idx_x_min = dr.arange(mi.UInt32, active_node_count) # Skip 0
        idx_x_max = dr.arange(mi.UInt32, 1, active_node_count + 1)

        dr.scatter(aabb_buffer, dr.gather(mi.Float, x, idx_x_min), idx * 6 + 0)
        dr.scatter(aabb_buffer, dr.gather(mi.Float, x, idx_x_max), idx * 6 + 3)

        points_idx = dr.binary_search(
            0, active_node_count,
            lambda index: dr.gather(mi.Float, x, index) < points.x
        ) - 1
        active = dr.neq(points_idx, INVALID)

        dr.eval(aabb_buffer, aabb_buffer_next, points_idx, leaves, active, leaves_count)
        log_runtime("Runtime:\t", log)

        loop_iter = 0
        while True:
            loop_iter += 1

            log_runtime("===============================", log, initial=True)

            #########################################################################
            # Split each node (`aabb_buffer`) into 8 sub-nodes (`aabb_buffer_next`) #
            #########################################################################
            idx = dr.arange(mi.UInt32, active_node_count)
            aabb_min, aabb_max = OcSpaceDistr.aabbs(aabb_buffer, idx)
            aabb_middle = (aabb_min + aabb_max) * 0.5
            OcSpaceDistr.split(
                aabb_buffer_next, aabb_min, aabb_max, aabb_middle, idx)
            log_runtime("    !& node split:\t\t\t", log)

            #########################################
            # Update `points_idx` for new sub-nodes #
            #########################################
            aabb_middle_pt = dr.gather(mi.Point3f, aabb_middle, points_idx, active)
            x_offset = dr.select(points.x > aabb_middle_pt.x, 1, 0)
            y_offset = dr.select(points.y > aabb_middle_pt.y, 1, 0)
            z_offset = dr.select(points.z > aabb_middle_pt.z, 1, 0)
            offset = OcSpaceDistr.split_offset(x_offset, y_offset, z_offset)
            active_node_count_opaque = dr.opaque(mi.UInt32, active_node_count)
            points_idx[active] = points_idx + offset * active_node_count_opaque
            log_runtime("    !& points_idx update for split:\t", log)

            ####################################
            # Determine which nodes are leaves #
            ####################################
            points_count = dr.zeros(mi.UInt32, active_node_count * 8)
            ones = dr.ones(mi.UInt32, dr.width(points))
            dr.scatter_reduce(dr.ReduceOp.Add, points_count, ones, points_idx, active)
            is_leaf = (
                ((loop_iter >= 3) & (points_count <= 1)) |
                (loop_iter >= self.max_depth)
            )
            log_runtime("    !& node termination:\t\t", log)

            #########################
            # Updates leaves buffer #
            #########################
            new_leaf_count = dr.count(is_leaf)
            if new_leaf_count > 0:
                leaf_idx = dr.compress(is_leaf)
                leaf_aabb_min = dr.gather(mi.Point3f, aabb_buffer_next, leaf_idx * 2 + 0)
                leaf_aabb_max = dr.gather(mi.Point3f, aabb_buffer_next, leaf_idx * 2 + 1)

                idx = dr.arange(mi.UInt32, new_leaf_count)
                dr.scatter(leaves, leaf_aabb_min, leaves_count * 2 + idx * 2 + 0)
                dr.scatter(leaves, leaf_aabb_max, leaves_count * 2 + idx * 2 + 1)

                leaves_count += dr.opaque(mi.UInt32, new_leaf_count)
            log_runtime("    !& leaf updates:\t\t\t", log)

            #############################################
            # Update `aabb_buffer` (for next iteration) #
            #############################################
            new_active_node_count = active_node_count * 8 - new_leaf_count
            if new_active_node_count > 0:
                idx = dr.compress(~is_leaf)
                aabb_next_min = dr.gather(mi.Point3f, aabb_buffer_next, idx * 2 + 0)
                aabb_next_max = dr.gather(mi.Point3f, aabb_buffer_next, idx * 2 + 1)

                idx = dr.arange(mi.UInt32, new_active_node_count)
                dr.scatter(aabb_buffer, aabb_next_min, idx * 2 + 0)
                dr.scatter(aabb_buffer, aabb_next_max, idx * 2 + 1)
            log_runtime("    !& buffer update:\t\t\t", log)

            ######################################
            # Update `points_idx` for new buffer #
            ######################################
            point_in_leaf = dr.gather(mi.Mask, is_leaf, points_idx, active)

            # Invalidate points that are now in leaves
            points_idx[active & point_in_leaf] = INVALID
            active &= dr.neq(points_idx, INVALID)

            is_new_leaf_int = dr.zeros(mi.UInt32, active_node_count * 8)
            is_new_leaf_int[is_leaf] = 1
            cumulative_new_leaf_count = dr.cumsum(is_new_leaf_int)
            leaf_count_before_point_idx = dr.gather(mi.UInt32, cumulative_new_leaf_count, points_idx, active)
            points_idx[active & (~point_in_leaf)] = points_idx - leaf_count_before_point_idx

            active_node_count = new_active_node_count
            log_runtime("    !& points_idx update for next:\t", log)

            ###########################
            # Early exit & evaluation #
            ###########################
            dr.eval(aabb_buffer, aabb_buffer_next, points_idx, leaves, active, leaves_count)
            leaves_count_scalar = dr.slice(leaves_count)
            log_runtime("    !& Evaluations for next iteration:\t", log)
            if log:
                print(f"l{loop_iter:2d}: ttl_leaf = {leaves_count_scalar:6d}, N_leaf_new = {new_leaf_count:6d}, N_node_new = {new_active_node_count:6d}")
            if active_node_count == 0:
                break
            if leaves_count_scalar + active_node_count * 8 > self.max_leaf_count:
                raise RuntimeError("OcSpaceDistr: Number of leaf nodes exceeds 'max_leaf_count'. Please increase 'max_leaf_count', increase 'octree_contruction_thres', or increase 'octree_construction_mean_mult'.")

        # Finalize
        del (aabb_buffer, aabb_buffer_next, points_idx, is_new_leaf_int,
             cumulative_new_leaf_count, leaf_count_before_point_idx, point_in_leaf,
             active, active_node_count, new_active_node_count, new_leaf_count)

        idx = dr.arange(mi.UInt32, leaves_count_scalar)
        aabb_min = dr.gather(mi.Point3f, leaves, idx * 2 + 0)
        aabb_max = dr.gather(mi.Point3f, leaves, idx * 2 + 1)

        return aabb_min, aabb_max


    def estimate_mass_in_leaves(self, aabb_min, aabb_max, seed, log:bool=False):
        """
        Evaluates `extra_spc` random samples in each leaf to compute an average
        mass per leaf. This average mass is then returned in a 1D PMF.
        """
        log_runtime(">>>>>>>>>>>>>>>>>>>>", log, initial=True)
        leaf_count = dr.width(aabb_min)  # TODO: opaque

        samples_count = self.extra_spc
        query_point = dr.zeros(mi.Point3f, leaf_count * samples_count)
        aabb_min_rep = dr.repeat(aabb_min, samples_count)
        aabb_max_rep = dr.repeat(aabb_max, samples_count)
        if True:
            sampler_extra = mi.load_dict({"type": "independent"})
            sampler_extra.seed(0xffffffff ^ seed, leaf_count * samples_count)
            query_point.x = dr.lerp(aabb_min_rep.x, aabb_max_rep.x, sampler_extra.next_1d())
            query_point.y = dr.lerp(aabb_min_rep.y, aabb_max_rep.y, sampler_extra.next_1d())
            query_point.z = dr.lerp(aabb_min_rep.z, aabb_max_rep.z, sampler_extra.next_1d())
        else:
            ri = mi.RadicalInverse(5, int(seed & 0))  # max_base, scramble
            query_num = dr.arange(mi.UInt32, 0, samples_count)
            sample_1 = dr.tile(ri.eval(0, query_num), leaf_count)  # base_index, num
            sample_2 = dr.tile(ri.eval(1, query_num), leaf_count)
            sample_3 = dr.tile(ri.eval(2, query_num), leaf_count)
            query_point.x = dr.lerp(aabb_min_rep.x, aabb_max_rep.x, sample_1)
            query_point.y = dr.lerp(aabb_min_rep.y, aabb_max_rep.y, sample_2)
            query_point.z = dr.lerp(aabb_min_rep.z, aabb_max_rep.z, sample_3)

        value, _ = self.eval_indirect_integrand_handle(query_point, sampler_extra)

        mass = dr.zeros(mi.Float, leaf_count)
        scatter_idx = dr.arange(mi.UInt32, leaf_count)
        scatter_idx = dr.repeat(scatter_idx, samples_count)
        dr.scatter_reduce(dr.ReduceOp.Add, mass, dr.max(value), scatter_idx)
        mass /= float(samples_count)

        log_runtime("Estimate_mass (extra sample): ", log, print_kernel=False)

        return mass


    def set_points(self, points, mass, seed=0, log=False):
    
        ######################################################
        # Preprocess input points (+ optional mass clamping) #
        ######################################################

        if not self.scatter_inc:
            # Launch a kernel to evaluate the points and mass. The results are
            # stored in memory for preprocess and optional clamping.
            dr.schedule(points)

            # Remove points with invalid mass
            valid_mask = mass > self.mass_contruction_thres
            valid_idx = dr.compress(valid_mask)
            points_ = dr.gather(mi.Point3f, points, valid_idx)
            if self._DEBUG_LOG:
                print(f"OcSpaceDistr: contructing octree with "
                      f"{dr.count(valid_mask)} valid points.")

            del points, mass, valid_idx
        else:
            valid_mask = mass > self.mass_contruction_thres
            if self._DEBUG_LOG:
                print(f"OcSpaceDistr: contructing octree with "
                      f"{dr.count(valid_mask)} valid points.")

            counter = mi.UInt32(0)
            compact_idx = dr.scatter_inc(
                target=counter,
                index=mi.UInt32(0),
                active=valid_mask)

            points_ = dr.zeros(mi.Point3f, self.max_leaf_count - 1)
            dr.scatter(
                target=points_,
                value=points,
                index=compact_idx,
                active=valid_mask & (compact_idx < self.max_leaf_count - 1)
            )

            # TODO: If there are more valid points than the max leaf count, no warning
            # is printed for efficiency. This usually leads to bias.
            # print(f"counter: {dr.count(valid_mask)} max_leaf_count: {self.max_leaf_count}")

            del points, mass, valid_mask, compact_idx

            # TODO: remove trailing unused points (is this necessary?) -> YES

        #######################
        # Octree construction #
        #######################

        _OCTREE_TIMINGS = False
        if _OCTREE_TIMINGS:
            kernel_history_flag_before = dr.flag(dr.JitFlag.KernelHistory)
            dr.set_flag(dr.JitFlag.KernelHistory, True)
            dr.kernel_history()
            time1 = time.time()

        lower, upper = self.construct_octree(points_, log=log)

        if _OCTREE_TIMINGS:
            time2 = time.time()
            full_runtime = (time2 - time1) * 1000
            print(f"  construct_octree: {full_runtime:.3f} [ms]")
            kernels = get_kernels(only_jit=False)
            kernel_runtime = sum([kernel['execution_time'] for kernel in kernels])
            kernel_compilation = sum([kernel['backend_time'] for kernel in kernels])
            print(f"\t\tSum of actual kernel runtimes: {kernel_runtime:.3f} [ms]")
            print(f"\t\tSum of kernel kernel compilations: {kernel_compilation:.3f} [ms]")
            print(f"\t\tLeaf nodes: {dr.width(lower)}")
            dr.set_flag(dr.JitFlag.KernelHistory, kernel_history_flag_before)

        ######################################################
        # Additional sampling per leaf (1D PMF construction) #
        ######################################################

        query_mass = self.estimate_mass_in_leaves(lower, upper, seed, log=log)

        # Store a discrete distribution of the partitioned space
        vol = (upper.x - lower.x) * (upper.y - lower.y) * (upper.z - lower.z)
        box_mass = query_mass * vol

        # Optional tradeoff between variance and (negligible) bias: Regions
        # whose contribution is many magnitudes smaller than others are very
        # rarely sampled in the rendering phase. But once sampled, they
        # contribute a lot to the variance since the reciprocal probability is
        # huge (>1e8). We can clamp the mass of these regions to suppress these
        # fireflies. This can be turned off if we have a large sampling budget
        # during rendering.
        if self.clamp_mass_thres > 0:
            query_mass_ttl = dr.sum(query_mass)
            thres = self.clamp_mass_thres * query_mass_ttl
            clamp_mask = query_mass < thres
            box_mass[clamp_mask] = 0
            if self._DEBUG_LOG:
                print(f"OcSpaceDistr: box mass clamped with threshold {thres} (ttl mass = {query_mass_ttl}) [{dr.width(clamp_mask)-dr.count(clamp_mask)}/{dr.width(clamp_mask)}]")
            valid_mask = box_mass > 0

            dr.schedule(valid_mask, lower, upper, box_mass)
            valid_idx = dr.compress(valid_mask)
            lower = dr.gather(mi.Point3f, lower, valid_idx)
            upper = dr.gather(mi.Point3f, upper, valid_idx)
            box_mass = dr.gather(mi.Float, box_mass, valid_idx)

            del valid_idx, valid_mask

        if self.sqrt_scale_mass:
            box_mass = dr.safe_sqrt(box_mass)

        del vol, query_mass, points_

        self.pmf = mi.DiscreteDistribution(box_mass)
        self.lower = lower
        self.upper = upper


    def sample(self, sampler):
        #idx, prob = self.pmf.sample_pmf(sampler.next_1d(), search_width=int(2e7))  # TODO: what number to use?
        idx, prob = self.pmf.sample_pmf(sampler.next_1d())  # TODO: what number to use?
        rcp_prob = dr.rcp(prob)

        p1 = dr.gather(mi.Point3f, self.lower, idx)
        p2 = dr.gather(mi.Point3f, self.upper, idx)
        sample_new = dr.zeros(mi.Vector3f, dr.width(p1))
        sample_new.x = dr.lerp(p1.x, p2.x, sampler.next_1d())
        sample_new.y = dr.lerp(p1.y, p2.y, sampler.next_1d())
        sample_new.z = dr.lerp(p1.z, p2.z, sampler.next_1d())
        vol = (p2.x - p1.x) * (p2.y - p1.y) * (p2.z - p1.z)
        rcp_pdf = vol * rcp_prob
        return sample_new, rcp_pdf

    def __repr__(self) -> str:
        return f"OcSpaceDistr []"
