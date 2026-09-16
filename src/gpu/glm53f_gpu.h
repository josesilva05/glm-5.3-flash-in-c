/* glm53f_gpu.h - CUDA backend: the trunk on one or more NVIDIA GPUs, routed experts on the CPU.
 *
 * WHAT RUNS WHERE
 *   GPU  mHC sites, KDA (projections, short conv, gates, delta-rule recurrence, gated norm),
 *        MLA (projections, KV cache, causal attention), dense MLPs, the shared expert of every
 *        MoE layer, final norm and lm_head. The KDA recurrent state and the MLA KV cache live
 *        on the GPU that owns the layer.
 *   CPU  embedding rows, the router and routed experts of every MoE layer (through the
 *        expert cache and its predictive prefetch), greedy argmax.
 *
 *   Per MoE layer only the normalised MoE input goes to the CPU and the routed sum comes
 *   back: two [T][hidden] transfers. While the CPU computes the routed experts the GPU
 *   computes the shared expert.
 *
 * NUMERICS
 *   Kernels repeat the CPU kernels' arithmetic: the same double accumulators, reduction
 *   trees and explicit fused multiply-adds (device code is compiled without automatic FMA
 *   contraction). Results can still differ from the CPU path in the last bits where a math
 *   library function (exp, tanh) rounds differently; they are validated against the same
 *   transformers reference as the CPU path.
 *
 * MULTIPLE GPUS
 *   Layers are assigned in order to the devices, each filled up to its free memory minus a
 *   reserve for its scratch, KV cache and state; lm_head goes on the last device used. The
 *   residual streams move between devices once per forward at each boundary.
 */
#ifndef GLM53F_GPU_H
#define GLM53F_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

struct Glm53fModel;
typedef struct Glm53fGpu Glm53fGpu;

/* Number of CUDA devices, 0 if none or if the build has no CUDA. */
int  glm53f_gpu_device_count(void);

/* Upload the model's trunk to the given devices (ndev <= 0: all). cap positions of KV cache.
 * Prints the placement. Returns NULL (with a message) if the trunk does not fit. */
Glm53fGpu *glm53f_gpu_create(struct Glm53fModel *m, const int *devices, int ndev);
void       glm53f_gpu_free(Glm53fGpu *g);

/* Same contract as glm53f_model_forward. */
int  glm53f_gpu_forward(Glm53fGpu *g, struct Glm53fModel *m, const int *ids, int T,
                        float *logits, int *argmax_all);

/* Zero the carried KDA state (the KV cache is overwritten as positions are reused). */
void glm53f_gpu_reset(Glm53fGpu *g);

#ifdef __cplusplus
}
#endif

#endif /* GLM53F_GPU_H */
