# MindNLP VLM 推理优化

## 概述

针对 MindNLP 框架下的 **Qwen2-VL** 和 **Janus-Pro** 模型进行了深度的推理性能优化。

优化核心在于将部分算子至 **MindSpore Mint** 算子体系、重构**图像预处理流水线**，以及优化 Attention 和 RoPE的计算逻辑。
显著降低端到端推理延迟，特别是提升多模态模型的首字（Prefill）响应速度。

## 核心优化内容

### 1\. 算子迁移 (Operator Optimization)

  * **Mint 算子全面替换:**
      * 将大量遗留的 tensor方法的`mindspore.ops` 算子替换为高性能的 `mindspore.mint` 算子（如 `mint.split`, `mint.cat`, `mint.transpose`, `mint.matmul`, `mint.softmax`, `mint.narrow` 等）。
      * **收益：** `mint` 算子能更好地映射到底层 Ascend 硬件算子，有着更快的运行速度，减少了框架层的算子分发开销。

### 2\. 高性能图像预处理 (High-Performance Image Preprocessing)

  * **从 PIL/NumPy 迁移至 MindData Vision:**
      * 重构了 `Qwen2VLImageProcessor` 和 `Janus` 的 `VLMImageProcessor`。
      * 将原本基于 Python/NumPy 的 `Resize`、`Rescale`、`Normalize` 操作，替换为 `mindspore.dataset.vision` 下的 C++ 后端算子。
      * **收益：** 解决了高分辨率图像处理中的 CPU 瓶颈，大幅缩短了多模态输入时的**首字延迟（First Token Latency）**。

### 3\. 模型架构层面的优化 (Model Architecture Optimizations)

#### **Janus-Pro优化**

  * **RoPE (旋转位置编码) 优化:**
      * 优化了 `apply_rotary_pos_emb` 和 `rotate_half` 函数。
      * 移除了不必要的 Tensor 切片与拼接操作，部分实现采用了 `mint.split` 或优化后的广播机制，提升了位置编码的计算效率。
  * **线性投影层重构:**
      * 在 `LlamaMLP` 和 `LlamaAttention` 中，优化了 `q_proj`, `k_proj`, `v_proj` 及其权重的分割逻辑，减少了冗余的内存拷贝。

#### **Qwen2-VL优化**

  * **3D 卷积优化:**
      * 将视频/图像 Patch Embedding 中的 `Conv3d` 实现替换为 `mindspore.mint.nn.functional.conv3d`，直接调用底层高性能算子。
  * **Attention Mask 计算优化:**
      * 移除了低效的 Python 循环生成 Mask 的逻辑。
      * 改用向量化的 `mint.arange`、`mint.where` 和 `mint.narrow` 进行 Mask 生成与切片，避免了大显存的占用和 host 侧的计算耗时。
    ```python
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights = ops.matmul(query_states, mindspore.mint.swapaxes(key_states, 2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = ops.matmul(attn_weights, value_states)
        attn_output = mindspore.mint.swapaxes(attn_output, 1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
    ```
  * **多模态 RoPE 优化:**
      * 预计算 Sin/Cos 表，避免在前向传播中重复计算；优化了多模态位置编码的广播（Broadcast）逻辑。
    ```python
        def apply_multimodal_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            q_embed = mindspore.ops.rotary_position_embedding(q, cos, sin, mode=0)
            k_embed = mindspore.ops.rotary_position_embedding(k, cos, sin, mode=0)
            return q_embed, k_embed
    ```

### 4\. 生成流程优化 (Generation Pipeline)

  * **停止条件判断 (Stopping Criteria):**
      * 将 `generation/utils.py` 中判断 `unfinished_sequences` 的逻辑从标量同步修改为向量化操作 (`unfinished_sequences.max() == 0`)，减少了 Device 到 Host 的同步等待时间。

## 性能收益

应用后，预期在 Ascend 硬件上获得以下提升：

1.  **更低的推理延迟 (Lower Latency):** 得益于 Mint 算子，单 Token 生成时间（Decode Latency）显著减少。
2.  **更快的首字响应 (Faster Prefill):** 图像预处理迁移至 C++ 后端后，VLM 模型的图片编码耗时大幅降低。


## 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 116.6667 |
| Prefill时延得分 | 424.8790    |
| Decode时延得分 | 238.1386     |
| **总分** | **259.8948** |