Of course. This is an excellent idea. The core philosophy of Plenoxels (direct, explicit optimization of a grid) combined with the geometric intuition of the divergence theorem is a powerful combination.

Here is a repurposed and comprehensive coding plan for an agent working on the Plenoxels codebase.

### **0. CLI Wrapper and Output Scheme (Immediate Step)**

Add a thin wrapper `scripts/run.py` to support the command:

- `python scripts/run.py --scene /path/to/.../transforms_train.json --network configs/nerf/base.json --n_steps 1000 --method surface --name hashsurfnew [--stride 25]`

Behavior:
- Converts `--scene` to the dataset root by taking the parent directory of `transforms_train.json`.
- Derives `dataset` and `scene` as the two directory names above the json file (e.g., `nerf_synthetic/drums`).
- Sets training output dir to `outputs/<dataset>/<scene>/<method>/<name>` and invokes the existing training entrypoint with `-t` pointing there, `data_dir` = dataset root, and `--config` from `--network`, `--n_iters` from `--n_steps`.
- After training, runs test rendering from the same output checkpoint using a custom `--stride` (default 25) on test views detected by replacing `transforms_train.json` with `transforms_test.json`.
- Push changes to fork after every major step.

### **Project Goal: Surface-Aware Plenoxels via Vector Potential Fields**

The objective is to adapt the Plenoxels framework to learn a *vector potential field* instead of direct spherical harmonic (SH) coefficients. The final appearance will be determined by the flux of this potential field through the local surface, approximated by a dot product with the surface normal derived from the density grid. This aims to embed geometric awareness directly into the appearance representation, potentially improving surface detail and view consistency.

### **Conceptual Overview**

1.  **Standard Plenoxel:** A sparse voxel grid stores scalar opacity `σ` and a vector of SH coefficients at each voxel corner. For a sample point, these values are trilinearly interpolated, and the resulting SH coefficients are evaluated with the view direction to produce a color.
2.  **Proposed Vector Potential Plenoxel (VPP):** The sparse grid will store scalar opacity `σ` and a *vector potential* for the SH coefficients (i.e., for each SH coefficient, we store a 3D vector). At a sample point, the rendering process is:
    a. Trilinearly interpolate the opacity `σ` and the SH vector potential **V_sh**.
    b. Calculate the local surface normal **n** by computing the gradient of the interpolated opacity field (`∇σ`).
    c. Compute the effective SH coefficients for rendering via a dot product: `SH_coeffs_eff = V_sh ⋅ n`.
    d. Evaluate these effective SH coefficients with the view direction to produce the final color.

---

### **Detailed Coding Plan**

#### **Phase 1: Setup and Data Structure Modification**

1.  **Branching Strategy:**
    *   Create a new git branch from the main Plenoxels codebase: `feature/vector-potential-plenoxels`.

2.  **Model Configuration:**
    *   Modify the argument parsing script to include a new model type. This allows for easy switching between the original implementation and your new version.
    *   **File to Modify:** `opt.py` or equivalent config script.
    *   **Example Arguments:**
        *   `--model_type=plenoxel` (Baseline)
        *   `--model_type=vector_potential` (New method)
        *   `--baseline_sh_multiplier=3` (For the fair comparison model)

3.  **Grid Data Representation:**
    *   The core change is in the width of the data array that the sparse grid indices point to. Let `K = (sh_degree + 1)**2` be the number of SH coefficients per color channel.
    *   **File to Modify:** The CUDA kernel file responsible for the grid structure (`*.cu`) and its corresponding Python wrapper (likely in `models.py` or a custom extension file).
    *   **Data Array Width:**
        *   **Baseline Plenoxel:** `1 (opacity) + 3 (RGB) * K (coeffs)`
        *   **Vector Potential Plenoxel:** `1 (opacity) + 3 (RGB) * K (coeffs) * 3 (vector components)`
    *   Adjust the memory allocation and indexing logic in both the Python host code and the CUDA device code to account for this new data width.

#### **Phase 2: Implementing the Core VPP Rendering Logic**

This phase is critical and likely involves changes to the custom CUDA kernels for performance. **It is highly recommended to first create a slower, PyTorch-native version to verify correctness before optimizing in CUDA.**

1.  **Density Gradient Calculation (Normal Vector):**
    *   This is the most crucial new component. For each sample point along a ray, you need to compute the gradient of the trilinearly interpolated density field.
    *   **Implementation Steps:**
        a. Within the rendering function (ideally the CUDA kernel), for a given sample point `p = (x, y, z)`, you must sample the density `σ` not only at `p` but also at neighboring points `p_x+ = (x+ε, y, z)`, `p_x- = (x-ε, y, z)`, etc., where `ε` is the voxel size.
        b. This requires performing 7 trilinear interpolations for density (`σ(p)` and its 6 neighbors) instead of one.
        c. Use the central difference formula to compute the gradient: `∇σ ≈ [ (σ(p_x+) - σ(p_x-))/(2ε), ... ]`.
        d. Normalize the resulting vector to get the normal `n = ∇σ / (||∇σ|| + 1e-8)`.

2.  **Vector Potential Dot Product:**
    *   After interpolating the SH vector potential features at the sample point, perform the dot product.
    *   **Pseudocode (within the rendering logic for one sample):**
        ```c++
        // In CUDA kernel, for a single sample point
        // 1. Calculate normal_vec (float3) using finite differences on the density grid.
        
        // 2. Interpolate the SH vector potential.
        // This will be a flat array of size 3 * K * 3.
        float* interpolated_sh_vp = ...; 

        // 3. Compute effective SH coefficients.
        float effective_sh_coeffs[3 * K];
        for (int c = 0; c < 3; ++c) { // R, G, B
            for (int i = 0; i < K; ++i) {
                int base_idx = (c * K + i) * 3;
                float3 potential_vec = make_float3(
                    interpolated_sh_vp[base_idx + 0],
                    interpolated_sh_vp[base_idx + 1],
                    interpolated_sh_vp[base_idx + 2]
                );
                // Dot product
                effective_sh_coeffs[c * K + i] = dot(potential_vec, normal_vec);
            }
        }

        // 4. Use the existing SH evaluation function with `effective_sh_coeffs`.
        float3 color = evaluate_sh(sh_degree, effective_sh_coeffs, view_dir);
        ```

#### **Phase 3: Setting Up a Fair Comparison Baseline**

1.  **Modified Baseline Model:**
    *   The goal is to create a standard Plenoxel model with a comparable number of parameters to your VPP model.
    *   Set the data array width to be the same as the VPP model: `1 (opacity) + 3 * K * 3`. This means for each color channel, you are storing `3K` "appearance features" instead of `K` SH coefficients.

2.  **Feature Reduction:**
    *   The standard `evaluate_sh` function cannot take `3K` coefficients. You must reduce these features down to the `K` coefficients it expects.
    *   Since Plenoxels avoids neural networks, a learnable `nn.Linear` layer is not the idiomatic choice. A fixed, non-learnable projection could be used, but a small learnable layer is the most principled way to make use of the extra parameters.
    *   **Recommendation:** Add a minimal MLP (e.g., a single `torch.nn.Linear` layer with no activation) that is applied *after* interpolation in the PyTorch part of the model. This is a slight deviation from the "no neural networks" rule but is necessary for a fair comparison of parameter counts.
    *   **Implementation:**
        *   In the model's `__init__`, define `self.feature_reducer = torch.nn.Linear(3 * K, K)`.
        *   In the forward pass (if prototyping in PyTorch), after interpolating the `3 * K * 3` features, reshape and pass them through this reducer before calling the SH evaluation.
        *   This will require pulling the interpolated values from CUDA back to PyTorch, applying the layer, and then potentially passing them back, which will be slow but correct for a baseline comparison.

#### **Phase 4: Training and Evaluation**

1.  **Loss Function and Regularization:**
    *   The primary reconstruction loss (MSE) remains unchanged.
    *   The Total Variation (TV) regularization now needs to operate on the wider data array. Ensure the TV loss calculation in the CUDA kernel correctly iterates over all `1 + 3*K*3` feature dimensions, applying the appropriate weights.

2.  **Experiment Design:**
    *   **Datasets:** Use the standard synthetic NeRF datasets (`Lego`, `Drums`, `Ship`, etc.) as they feature distinct, opaque surfaces where this method is most likely to show an advantage.
    *   **Train both models:**
        1.  `--model_type=vector_potential`
        2.  `--model_type=plenoxel --baseline_sh_multiplier=3`
    *   Train them for the same number of iterations as the original Plenoxel paper.
    *   **Metrics & Analysis:**
        *   Compare PSNR, SSIM, and LPIPS.
        *   Pay close attention to qualitative results. Inspect high-frequency surface details, specular reflections, and object silhouettes. The hypothesis is that the VPP model may produce sharper, more geometrically consistent details.

### **Summary of Agent's Tasks**

1.  **Setup:** Branch the codebase and add `--model_type` and `--baseline_sh_multiplier` arguments.
2.  **Modify Data Structure:** Update the data array width in the Python and CUDA code to be `1 + 3 * K * 3` for the new models.
3.  **Implement VPP Rendering Core:**
    *   Prototype in PyTorch first for correctness.
    *   Implement a density gradient calculation function using finite differences on the interpolated grid.
    *   Implement the dot product between the interpolated vector potential and the normal to get effective SH coefficients.
    *   (Optional but recommended for performance) Port this logic into the existing CUDA rendering kernel.
4.  **Implement Fair Baseline:**
    *   Use the same wide data array as the VPP.
    *   Add a minimal `torch.nn.Linear` layer to reduce the `3K` features to `K` before SH evaluation.
5.  **Run Experiments:** Train and evaluate both models on synthetic datasets, comparing both quantitative metrics and qualitative visual results.