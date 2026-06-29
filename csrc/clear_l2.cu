// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <cuda/cmath>

#include "utils.h"

__global__ void write_cache_kernel(uint4* dummy_memory, const int size) {
    const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    dummy_memory[i] = make_uint4(0, 0, 0, 0);
}

__global__ void discard_cache_kernel(uint4* dummy_memory, const int size) {
    const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;

    asm volatile(
           "discard.global.L2 [%0], 128;"
           :
           :"l"(dummy_memory + i)
           :"memory");
}

// Neutralize any L2 *persistence* setup before the spam-write, so the write can actually
// evict everything. A submission can otherwise keep a buffer resident in L2 across timed
// iterations by marking it persisting via an access-policy window
// (cudaStreamSetAttribute, hitProp=cudaAccessPropertyPersisting) backed by a set-aside L2
// carveout (cudaLimitPersistingL2CacheSize). Streaming/normal writes can only use the
// *non*-set-aside portion of L2, so the write below cannot evict lines parked in the
// carveout -- the warm buffer then survives the "cache clear" and inflates the score.
//
// Note: on recent GPUs (e.g. Hopper/Blackwell) the default persisting carveout is already
// nonzero, so a submission only needs to set an access-policy window -- no explicit
// cudaDeviceSetLimit call -- to start parking data in the set-aside region.
static void disarm_l2_persistence(cudaStream_t stream) {
    // Release the set-aside carveout so the spam-write can use the whole L2 and evict
    // everything (including lines the previous kernel parked as persisting). Changing this
    // device limit drains in-flight work, which also guarantees the previous (untrusted)
    // kernel has finished before we demote+evict its lines. Only pay that cost when a
    // carveout is actually reserved: honest kernels never set it, so after the first clear
    // the limit stays 0 and this branch is skipped (cudaDeviceGetLimit is cheap/non-blocking).
    std::size_t persist_limit = 0;
    if (cudaDeviceGetLimit(&persist_limit, cudaLimitPersistingL2CacheSize) == cudaSuccess &&
        persist_limit != 0) {
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
    }

    // Demote any lines still flagged persisting to normal status so the write can evict them.
    CUDA_CHECK(cudaCtxResetPersistingL2Cache());

    // Drop any access-policy window the submission left on the benchmark stream, so the next
    // kernel launch does not immediately re-mark its inputs persisting and re-pin them.
    cudaStreamAttrValue window_attr;
    std::memset(&window_attr, 0, sizeof(window_attr));
    window_attr.accessPolicyWindow.num_bytes = 0;
    CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &window_attr));
}

void clear_cache(void* dummy_memory, int size, bool discard, cudaStream_t stream) {
    // make sure there's no sneaky cache persistency setting that defeats our spam-writing
    disarm_l2_persistence(stream);

    int nelem = size / sizeof(uint4);
    // write a large amount of memory to ensure all cache lines are cleared
    int threads = 256;
    int blocks = cuda::ceil_div(nelem, threads);
    write_cache_kernel<<<blocks, threads, 0, stream>>>(static_cast<uint4*>(dummy_memory), nelem);
    CUDA_CHECK(cudaGetLastError());
    if (discard) {
        discard_cache_kernel<<<blocks, threads, 0, stream>>>(static_cast<uint4*>(dummy_memory), nelem);
        CUDA_CHECK(cudaGetLastError());
    }
}
