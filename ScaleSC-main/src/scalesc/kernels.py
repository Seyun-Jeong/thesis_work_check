from __future__ import annotations

import cupy as cp
from cuml.common.kernel_utils import cuda_kernel_factory

get_mean_var_major_kernel = r"""
        (const int *indptr,const int *index,const {0} *data,
            double* means,double* vars,
            int major, int minor) {
        int major_idx = blockIdx.x;
        if(major_idx >= major){
            return;
        }
        int start_idx = indptr[major_idx];
        int stop_idx = indptr[major_idx+1];

        __shared__ double mean_place[64];
        __shared__ double var_place[64];

        mean_place[threadIdx.x] = 0.0;
        var_place[threadIdx.x] = 0.0;
        __syncthreads();

        for(int minor_idx = start_idx+threadIdx.x; minor_idx < stop_idx; minor_idx+= blockDim.x){
               double value = (double)data[minor_idx];
               mean_place[threadIdx.x] += value;
               var_place[threadIdx.x] += value*value;
        }
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                mean_place[threadIdx.x] += mean_place[threadIdx.x + s];
                var_place[threadIdx.x] += var_place[threadIdx.x + s];
            }
            __syncthreads(); // Synchronize at each step of the reduction
        }
        if (threadIdx.x == 0) {
            means[major_idx] = mean_place[threadIdx.x];
            vars[major_idx] = var_place[threadIdx.x];
        }

        }
"""

"""
    Modified: return sum(x) and sq_sum(x)
"""
get_mean_var_minor_kernel = r"""
        (const int *index,const {0} *data,
            double* sums, double* sq_sums,
            int major, int nnz) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx >= nnz){
            return;
        }
        double value = (double) data[idx];
        int minor_pos = index[idx];
        atomicAdd(&sums[minor_pos], value);
        atomicAdd(&sq_sums[minor_pos], value*value);
        }
    """

find_indices_kernel = r"""
extern "C" __global__ void find_indices(const long int* A, const long int* indptr, int* result, long int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        long int value = A[idx];
        long int left = 0;
        long int right = M - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (indptr[mid] <= value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (left > 0 && indptr[left - 1] <= value && left < M) {
            result[idx] = left - 1; 
        } else {
            result[idx] = -1; 
        }
    }
}
"""

check_in_cols_kernel = cp.ElementwiseKernel(
    'int32 index, int32 cols_size, raw int32 cols',
    'bool is_in',
    '''
    is_in = false;
    for (int i = 0; i < cols_size; i++) {
        if (index == cols[i]) {
            is_in = true;
            break;
        }
    }
    ''',
    'check_in_cols'
)


csr_row_index_kernel = cp.ElementwiseKernel(
    'int32 out_rows, raw I rows, '
    'raw int64 Ap, raw int32 Aj, raw T Ax, raw int64 Bp',
    'int32 Bj, T Bx',
    '''
    const I row = rows[out_rows];

    // Look up starting offset
    const I starting_output_offset = Bp[out_rows];
    const I output_offset = i - starting_output_offset;
    const I starting_input_offset = Ap[row];

    Bj = Aj[starting_input_offset + output_offset];
    Bx = Ax[starting_input_offset + output_offset];
''', 'cupyx_scipy_sparse_csr_row_index_ker')



sq_sum = cp.ReductionKernel(
    "T x",  # input params
    "float64 y",  # output params
    "x * x",  # map
    "a + b",  # reduce
    "y = a",  # post-reduction map
    "0",  # identity value
    "sqsum64",  # kernel name
)

mean_sum = cp.ReductionKernel(
    "T x",  # input params
    "float64 y",  # output params
    "x",  # map
    "a + b",  # reduce
    "y = a",  # post-reduction map
    "0",  # identity value
    "sum64",  # kernel name
)


seurat_v3_elementwise_kernel = cp.ElementwiseKernel(
    "T data, S idx, raw D clip_val",
    "raw D sq_sum, raw D sum",
    """
    D element = min((double)data, clip_val[idx]);
    atomicAdd(&sq_sum[idx], element * element);
    atomicAdd(&sum[idx], element);
    """,
    "seurat_v3_elementwise_kernel",
    no_return=True,
)


sum_sign_elementwise_kernel = cp.ElementwiseKernel(
    "T data, S idx",
    "raw D sum",
    """
    if (data > 0){
        atomicAdd(&sum[idx], 1);
    }else{
        atomicAdd(&sum[idx], 0);
    }
    """,
    "sum_sign_elementwise_kernel",
    no_return=True,
)

def get_mean_var_major(dtype):
    return cuda_kernel_factory(
        get_mean_var_major_kernel, (dtype,), "get_mean_var_major_kernel"
    )


def get_mean_var_minor(dtype):
    return cuda_kernel_factory(
        get_mean_var_minor_kernel, (dtype,), "get_mean_var_minor_kernel"
    )


def get_find_indices():
    return cp.RawKernel(find_indices_kernel, 'find_indices')

