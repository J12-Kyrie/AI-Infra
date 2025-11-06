# vLLM V0与V1核心组件深度解析

## 目录
1. [概述](#概述)
2. [V0核心组件深度解析](#v0核心组件深度解析)
3. [V1核心组件深度解析](#v1核心组件深度解析)
4. [核心原理对比](#核心原理对比)
5. [以Qwen2模型为例的推理流程](#以qwen2模型为例的推理流程)

---

## 概述

vLLM的核心组件是推理引擎的基础架构，负责调度、内存管理、模型执行等关键功能。V0和V1在这些核心组件上有显著的设计差异，本文档将深入分析两个版本的核心组件实现原理。

**核心组件层次结构**:
```
调度层(Scheduler) → 内存管理层(BlockManager/KVCacheManager) → 执行层(Worker) → 模型运行层(ModelRunner)
```

---

## V0核心组件深度解析

### 1. V0 Scheduler (core/scheduler.py)

#### 1.1 核心职责
- **请求队列管理**: 维护waiting、running、swapped三个队列
- **Prefill/Decode分离调度**: 分阶段处理prefill和decode请求
- **内存感知调度**: 根据KV cache可用性决定调度策略
- **抢占机制**: 支持SWAP和RECOMPUTE两种抢占模式

#### 1.2 核心数据结构

```python
# core/scheduler.py:426-520
class Scheduler:
    # 请求队列
    self.waiting: Deque[SequenceGroup] = deque()     # 等待队列
    self.running: Deque[SequenceGroup] = deque()     # 运行队列
    self.swapped: Deque[SequenceGroup] = deque()     # 换出队列

    # 内存管理器
    self.block_manager: BlockSpaceManager            # KV cache块管理

    # 调度约束
    self.max_num_batched_tokens: int                 # 最大批次token数
    self.max_num_seqs: int                           # 最大序列数
```

**SchedulingBudget** (line 49-124):
```python
@dataclass
class SchedulingBudget:
    token_budget: int                                # token预算
    max_num_seqs: int                                # 序列数上限
    _num_batched_tokens: int                         # 已批次化token数
    _num_curr_seqs: int                              # 当前序列数

    def can_schedule(self, num_new_tokens: int, num_new_seqs: int) -> bool:
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)
```

#### 1.3 调度算法原理

**默认调度算法** (_schedule_default, line 1230-1356):

```
调度优先级:
1. Prefill优先: 优先调度waiting队列中的新请求
2. Decode持续: 若无prefill，调度running队列中的decode请求
3. Swap-in: 若有空间，将swapped请求换入GPU
4. Preemption: 内存不足时抢占最低优先级请求

算法流程:
Step 1: 创建调度预算(SchedulingBudget)
    budget = SchedulingBudget(
        token_budget=max_num_batched_tokens,
        max_num_seqs=max_num_seqs
    )

Step 2: 调度Prefill请求(_schedule_prefills)
    FOR request IN waiting:
        计算 num_new_tokens = request.get_num_uncomputed_tokens()

        检查块分配:
            alloc_status = block_manager.can_allocate(request)

        IF alloc_status == OK AND budget.can_schedule(num_new_tokens, 1):
            block_manager.allocate(request)
            request.status = RUNNING
            running.append(request)
            budget.add_num_batched_tokens(num_new_tokens)

Step 3: 调度Decode请求(_schedule_running)
    IF no prefills scheduled:
        FOR request IN running:
            num_new_tokens = 1  # decode每次1个token

            IF block_manager.can_append_slots(request):
                block_manager.append_slots(request)
                scheduled_decode.append(request)
                budget.add_num_batched_tokens(1)
            ELSE:
                # 抢占
                victim = running.pop()
                preempt(victim)
                waiting.appendleft(victim)

Step 4: Swap-in请求(_schedule_swapped)
    IF no preemptions:
        FOR request IN swapped:
            IF block_manager.can_swap_in(request):
                block_manager.swap_in(request)
                request.status = RUNNING
                running.append(request)
```

**Chunked Prefill调度算法** (_schedule_chunked_prefill, line 1358-1471):

与默认调度的关键区别:
```
1. Decode优先: 先调度所有decode请求
2. Chunked Prefill: prefill可以与decode混合批次
3. 并发Prefill: 支持多个prefill请求同时运行
4. 动态分块: prefill可以分多次完成

调度顺序:
1. _schedule_running(enable_chunking=True)    # Decode + Chunked Prefill
2. _schedule_swapped()                        # Swap-in
3. _schedule_prefills(enable_chunking=True)   # 新Prefill请求
```

#### 1.4 抢占机制

**抢占策略** (_preempt, line 1776-1819):

```python
# 抢占模式选择逻辑
IF user_specified_preemption_mode is None:
    IF seq_group.get_max_num_running_seqs() == 1:
        preemption_mode = RECOMPUTE  # 单序列 → 重计算
    ELSE:
        preemption_mode = SWAP       # 多序列(beam search) → 换出
ELSE:
    preemption_mode = user_specified_preemption_mode

# RECOMPUTE抢占: 释放所有块，重置计算状态
IF preemption_mode == RECOMPUTE:
    FOR seq IN seq_group.get_seqs(RUNNING):
        seq.status = WAITING
        block_manager.free(seq)
        seq.reset_state_for_recompute()  # 重置num_computed_tokens

# SWAP抢占: 将GPU块换出到CPU
ELIF preemption_mode == SWAP:
    mapping = block_manager.swap_out(seq_group)
    blocks_to_swap_out.extend(mapping)
    FOR seq IN seq_group.get_seqs(RUNNING):
        seq.status = SWAPPED
```

**抢占时机**:
- 分配新块时内存不足 (line 739-787)
- 追加新slot时内存不足 (can_append_slots返回False)

#### 1.5 前缀缓存支持

```python
# 前缀缓存命中检测 (line 1579-1582)
IF cache_config.enable_prefix_caching:
    common_computed_block_nums = (
        block_manager.get_common_computed_block_ids(
            seq_group.get_seqs(RUNNING)))
    # 返回已计算的公共前缀块ID列表
```

---

### 2. V0 BlockSpaceManager (core/block_manager.py)

#### 2.1 核心职责
- **KV Cache块分配**: 管理GPU和CPU上的物理块
- **块表映射**: 维护逻辑块到物理块的映射
- **Swap操作**: CPU↔GPU块交换
- **前缀缓存**: 支持prefix caching优化

#### 2.2 核心数据结构

```python
# core/block_manager.py:22-109
class SelfAttnBlockSpaceManager(BlockSpaceManager):
    def __init__(
        self,
        block_size: int,                    # 块大小(e.g., 16 tokens)
        num_gpu_blocks: int,                # GPU块数
        num_cpu_blocks: int,                # CPU块数
        watermark: float = 0.01,            # 水位线(避免频繁换页)
        sliding_window: Optional[int] = None,  # 滑动窗口
        enable_caching: bool = False,       # 是否启用前缀缓存
    ):
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # 块分配器
        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        # 块表: seq_id -> BlockTable
        self.block_tables: Dict[SeqId, BlockTable] = {}

        # 前缀缓存追踪器
        self._computed_blocks_tracker = ComputedBlocksTracker(...)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(...)
```

**BlockTable**:
```python
# core/block/block_table.py
class BlockTable:
    """
    逻辑块表，维护一个序列的token到物理块的映射

    核心功能:
    - allocate(): 为新token分配块
    - append_token_ids(): 追加token到最后一个块或分配新块
    - fork(): 创建copy-on-write的子表
    - free(): 释放所有块
    """

    def __init__(
        self,
        block_size: int,
        block_allocator: BlockAllocator,
        max_block_sliding_window: Optional[int] = None,
    ):
        self.block_size = block_size
        self.block_allocator = block_allocator
        self.blocks: List[Block] = []  # 块列表
        self._num_full_slots = 0       # 已使用的slot数
```

#### 2.3 块分配算法

**can_allocate** (line 110-147):
```python
def can_allocate(
    seq_group: SequenceGroup,
    num_lookahead_slots: int = 0
) -> AllocStatus:
    """
    检查是否可以为seq_group分配块

    返回值:
        AllocStatus.OK: 可以立即分配
        AllocStatus.LATER: 内存不足但未来可能有空间
        AllocStatus.NEVER: 请求过大，永远无法分配
    """

    # 计算所需块数
    seq = seq_group.get_seqs(WAITING)[0]
    num_required_blocks = ceil(
        (len(seq.get_token_ids()) + num_lookahead_slots) / block_size)

    # 考虑滑动窗口限制
    IF max_block_sliding_window is not None:
        num_required_blocks = min(num_required_blocks, max_block_sliding_window)

    num_free_gpu_blocks = block_allocator.get_num_free_blocks(GPU)

    # 使用水位线避免频繁换页
    IF (num_total_gpu_blocks - num_required_blocks < watermark_blocks):
        RETURN AllocStatus.NEVER
    IF (num_free_gpu_blocks - num_required_blocks >= watermark_blocks):
        RETURN AllocStatus.OK
    ELSE:
        RETURN AllocStatus.LATER
```

**allocate** (line 166-206):
```python
def allocate(seq_group: SequenceGroup) -> None:
    """为seq_group的所有序列分配块"""

    waiting_seqs = seq_group.get_seqs(WAITING)
    seq = waiting_seqs[0]

    # 创建块表并分配块
    block_table = BlockTable(block_size, block_allocator, ...)
    IF seq.get_token_ids():
        # 计算哈希用于前缀缓存
        extra_hash = seq.extra_hash()
        block_table.allocate(
            token_ids=seq.get_token_ids(),
            extra_hash=extra_hash
        )

    self.block_tables[seq.seq_id] = block_table

    # 为其他序列fork块表 (beam search)
    FOR seq IN waiting_seqs[1:]:
        self.block_tables[seq.seq_id] = block_table.fork()
```

**append_slots** (line 236-252):
```python
def append_slots(
    seq: Sequence,
    num_lookahead_slots: int,
) -> List[Tuple[int, int]]:
    """
    为序列追加新slots

    返回: Copy-on-Write操作列表 [(src_block_id, dst_block_id)]
    """

    block_table = self.block_tables[seq.seq_id]

    # 获取未见过的token
    unseen_token_ids = block_table.get_unseen_token_ids(seq.get_token_ids())

    # 追加token (可能触发新块分配或CoW)
    block_table.append_token_ids(
        token_ids=unseen_token_ids,
        num_lookahead_slots=num_lookahead_slots,
        num_computed_slots=seq.data.get_num_computed_tokens(),
        extra_hash=seq.extra_hash(),
    )

    # 返回CoW操作
    new_cows = self.block_allocator.clear_copy_on_writes()
    RETURN new_cows
```

#### 2.4 Swap操作

**swap_in** (line 361-396):
```python
def swap_in(seq_group: SequenceGroup) -> List[Tuple[int, int]]:
    """
    将seq_group的块从CPU换入GPU

    返回: [(cpu_block_id, gpu_block_id), ...]
    """

    physical_block_id_mapping = []

    FOR seq IN seq_group.get_seqs(SWAPPED):
        blocks = self.block_tables[seq.seq_id].blocks
        IF len(blocks) == 0:
            CONTINUE

        # 执行swap: CPU → GPU
        seq_swap_mapping = block_allocator.swap(
            blocks=blocks,
            src_device=CPU,
            dst_device=GPU
        )

        # 更新块表中的块ID
        self.block_tables[seq.seq_id].update(blocks)

        # 构建物理块ID映射
        seq_physical_block_id_mapping = {
            block_allocator.get_physical_block_id(CPU, cpu_id):
            block_allocator.get_physical_block_id(GPU, gpu_id)
            FOR cpu_id, gpu_id IN seq_swap_mapping.items()
        }

        physical_block_id_mapping.extend(list(seq_physical_block_id_mapping.items()))

    RETURN physical_block_id_mapping
```

**swap_out** (line 414-449):
类似swap_in，但方向相反: GPU → CPU

#### 2.5 前缀缓存机制

**get_common_computed_block_ids** (line 310-333):
```python
def get_common_computed_block_ids(seqs: List[Sequence]) -> Sequence[int]:
    """
    获取所有序列共享的已计算块ID

    用于前缀缓存优化: 跳过已计算的公共前缀块
    """

    computed_seq_block_ids = []
    FOR seq IN seqs:
        all_blocks = self.block_tables[seq.seq_id].physical_block_ids

        # 获取已缓存的token数 (必须是block_size的倍数)
        num_cached_tokens = computed_blocks_tracker.get_num_cached_tokens(seq)
        ASSERT num_cached_tokens % block_size == 0

        num_cached_blocks = num_cached_tokens // block_size
        computed_block_ids = all_blocks[:num_cached_blocks]
        computed_seq_block_ids.append(computed_block_ids)

    # 返回所有序列的公共前缀块
    RETURN block_allocator.get_common_computed_block_ids(computed_seq_block_ids)
```

**ComputedBlocksTracker**:
```python
# core/block/prefix_caching_block.py
class ComputedBlocksTracker:
    """
    追踪哪些块已经被计算过 (用于前缀缓存)

    核心功能:
    - get_num_cached_tokens(seq): 获取序列的已缓存token数
    - 在第一次prefill调度时计算缓存命中
    - 对于chunked prefill，缓存第一次计算的token数
    """
```

---

### 3. V0 Worker (worker/worker.py)

#### 3.1 核心职责
- **模型管理**: 加载和管理GPU上的模型
- **KV Cache引擎**: 管理物理KV cache张量
- **执行协调**: 协调输入准备、KV操作、模型执行

#### 3.2 核心数据结构

```python
# worker/worker.py:39-130
class Worker(WorkerBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        ...
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config

        # 模型运行器
        self.model_runner: GPUModelRunnerBase = GPUModelRunner(...)

        # KV cache引擎
        self.cache_engine: List[CacheEngine] = []

        # GPU KV cache张量
        self.gpu_cache: List[List[torch.Tensor]] = []
```

#### 3.3 执行流程

**execute_model** (line 387-449):
```python
def execute_model(
    execute_model_req: Optional[ExecuteModelRequest] = None,
) -> Optional[List[SamplerOutput]]:
    """
    执行模型推理的主入口

    流程:
    1. 准备输入
    2. 执行Worker操作 (KV cache swap/copy)
    3. Pipeline并行: 接收/发送中间张量
    4. 执行模型forward
    5. 返回采样结果
    """

    # Step 1: 准备输入
    inputs = self.prepare_input(execute_model_req)
    model_input, worker_input, kwargs = inputs

    # Step 2: 执行Worker操作 (KV cache管理)
    self.execute_worker(worker_input)
    # 内部执行:
    # - cache_engine.swap_in(blocks_to_swap_in)
    # - cache_engine.swap_out(blocks_to_swap_out)
    # - cache_engine.copy(blocks_to_copy)

    # Step 3: Pipeline并行 - 接收中间张量 (非首rank)
    IF not get_pp_group().is_first_rank():
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict()
        )

    # Step 4: 执行模型
    output = self.model_runner.execute_model(
        model_input=model_input,
        kv_caches=self.kv_cache[worker_input.virtual_engine],
        intermediate_tensors=intermediate_tensors,
        num_steps=num_steps,
    )

    # Step 5: Pipeline并行 - 发送中间张量 (非末rank)
    IF not get_pp_group().is_last_rank():
        get_pp_group().send_tensor_dict(output.tensors)
        RETURN [None]

    # Step 6: 返回采样结果
    RETURN output  # List[SamplerOutput]
```

**KV Cache初始化** (determine_num_available_blocks):
```python
def determine_num_available_blocks() -> int:
    """
    通过profile run确定可用的KV cache块数

    流程:
    1. 执行一次完整的forward pass (profile run)
    2. 记录峰值内存使用
    3. 计算可用于KV cache的内存
    4. 根据块大小计算可分配的块数
    """

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 执行profile run
    self.model_runner.profile_run()

    # 计算可用内存
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    memory_for_kv_cache = (
        total_gpu_memory * gpu_memory_utilization
        - model_weight_memory
        - activation_memory
    )

    # 计算块大小
    cache_block_size = (
        block_size *              # e.g., 16
        num_kv_heads *           # e.g., 32
        head_size *              # e.g., 128
        num_layers *             # e.g., 32
        2 *                      # K和V
        dtype_size               # e.g., 2 bytes for fp16
    )

    num_gpu_blocks = memory_for_kv_cache // cache_block_size
    RETURN num_gpu_blocks
```

---

### 4. V0 ModelRunner (worker/model_runner.py)

#### 4.1 核心职责
- **输入准备**: 构建模型输入张量和attention metadata
- **模型执行**: 调用模型forward和采样
- **输出处理**: 处理logits和采样结果

#### 4.2 核心流程

**prepare_model_input**:
```python
def prepare_model_input(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    ...
) -> ModelRunnerInputBase:
    """
    准备模型输入

    构建:
    - input_ids: Token IDs [num_tokens]
    - positions: Position IDs [num_tokens]
    - attn_metadata: Attention元数据
    - slot_mapping: KV cache槽位映射 [num_tokens]
    """

    # 收集所有token和位置
    input_tokens = []
    input_positions = []
    slot_mapping = []

    FOR seq_group_meta IN seq_group_metadata_list:
        IF seq_group_meta.is_prompt:
            # Prefill阶段
            prompt_tokens = seq_group_meta.seq_data[seq_id].get_token_ids()
            prompt_len = len(prompt_tokens)

            input_tokens.extend(prompt_tokens)
            input_positions.extend(range(prompt_len))

            # 计算slot mapping
            block_table = seq_group_meta.block_tables[seq_id]
            FOR i, token IN enumerate(prompt_tokens):
                block_idx = i // block_size
                block_offset = i % block_size
                slot = block_table[block_idx] * block_size + block_offset
                slot_mapping.append(slot)
        ELSE:
            # Decode阶段
            last_token_id = seq_group_meta.seq_data[seq_id].get_last_token_id()
            input_tokens.append(last_token_id)

            seq_len = seq_group_meta.seq_data[seq_id].get_len()
            input_positions.append(seq_len - 1)

            # Slot mapping for decode
            block_idx = (seq_len - 1) // block_size
            block_offset = (seq_len - 1) % block_size
            block_table = seq_group_meta.block_tables[seq_id]
            slot = block_table[block_idx] * block_size + block_offset
            slot_mapping.append(slot)

    # 构建attention metadata
    attn_metadata = self.attn_backend.make_metadata(
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        slot_mapping=slot_mapping,
        ...
    )

    RETURN ModelRunnerInputBase(
        input_tokens=torch.tensor(input_tokens, device="cuda"),
        input_positions=torch.tensor(input_positions, device="cuda"),
        attn_metadata=attn_metadata,
        ...
    )
```

**execute_model**:
```python
def execute_model(
    model_input: ModelRunnerInputBase,
    kv_caches: List[torch.Tensor],
    ...
) -> List[SamplerOutput]:
    """
    执行模型forward和采样
    """

    # Step 1: 模型forward
    hidden_states = self.model(
        input_ids=model_input.input_tokens,
        positions=model_input.input_positions,
        kv_caches=kv_caches,
        attn_metadata=model_input.attn_metadata,
    )

    # Step 2: 计算logits
    logits = self.model.compute_logits(hidden_states, None)

    # Step 3: 采样
    sampler_output = self.sampler(
        logits=logits,
        sampling_metadata=model_input.sampling_metadata,
    )

    RETURN [sampler_output]
```

---

## V1核心组件深度解析

### 1. V1 Scheduler (v1/core/sched/scheduler.py)

#### 1.1 核心创新点

**统一Token级调度** (line 158-168):
```
V1调度器的核心设计理念:

没有"prefill阶段"和"decode阶段"的区分。
每个请求有两个关键属性:
- num_computed_tokens: 已计算的token数
- num_tokens_with_spec: 总需求token数 =
    len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)

调度目标: 让每个请求的 num_computed_tokens 追赶其 num_tokens_with_spec

这种设计统一支持:
1. Chunked prefills (分块prefill)
2. Prefix caching (前缀缓存)
3. Speculative decoding (投机解码)
4. Future: Jump decoding
```

#### 1.2 核心数据结构

```python
# v1/core/sched/scheduler.py:38-156
class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        ...
    ):
        # 请求队列 (简化: 只有waiting和running)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # req_id -> Request
        self.requests: dict[str, Request] = {}

        # 调度约束
        self.max_num_running_reqs = scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = scheduler_config.max_num_batched_tokens
        self.max_model_len = scheduler_config.max_model_len

        # KV cache管理器 (V1增强版)
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=cache_config.enable_prefix_caching,
            ...
        )

        # 编码器cache管理器 (V1新增)
        self.encoder_cache_manager = EncoderCacheManager(...)

        # 结构化输出管理器 (V1新增)
        self.structured_output_manager = structured_output_manager

        # KV Connector for P/D and offloading (V1新增)
        self.connector: Optional[KVConnectorBase_V1] = (
            KVConnectorFactory.create_connector_v1(...)
            if kv_transfer_config else None
        )
```

#### 1.3 统一调度算法

**schedule** (line 158-584):
```python
def schedule() -> SchedulerOutput:
    """
    V1统一调度算法

    核心思想: Token级调度，无阶段区分
    """

    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens

    # Step 1: 调度RUNNING请求
    req_index = 0
    WHILE req_index < len(self.running) AND token_budget > 0:
        request = self.running[req_index]

        # 计算需要调度的token数
        num_new_tokens = (request.num_tokens_with_spec -
                          request.num_computed_tokens)

        # 限制长prefill的chunk大小
        IF 0 < long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = long_prefill_token_threshold

        num_new_tokens = min(num_new_tokens, token_budget)

        # 确保不超过max_model_len
        num_new_tokens = min(
            num_new_tokens,
            max_model_len - request.num_computed_tokens)

        # 调度编码器输入 (如果是多模态)
        IF request.has_encoder_inputs:
            encoder_inputs_to_schedule, num_new_tokens, encoder_budget = (
                self._try_schedule_encoder_inputs(
                    request, request.num_computed_tokens,
                    num_new_tokens, encoder_budget))

        IF num_new_tokens == 0:
            req_index += 1
            CONTINUE

        # 计算投机token数量
        num_draft_tokens = max(
            num_new_tokens + request.num_computed_tokens - request.num_tokens,
            0)

        # 分配KV cache
        WHILE True:
            new_blocks = kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_draft_tokens=num_draft_tokens,
                num_lookahead_tokens=num_lookahead_tokens)

            IF new_blocks is None:
                # 内存不足，抢占最低优先级请求
                preempted_req = self.running.pop()
                kv_cache_manager.free(preempted_req)
                preempted_req.status = RequestStatus.PREEMPTED
                preempted_req.num_computed_tokens = 0

                waiting.appendleft(preempted_req)
                preempted_reqs.append(preempted_req)

                IF preempted_req == request:
                    can_schedule = False
                    BREAK
            ELSE:
                can_schedule = True
                BREAK

        IF not can_schedule:
            BREAK

        # 调度成功
        scheduled_running_reqs.append(request)
        req_to_new_block_ids[request.request_id] = new_blocks.get_block_ids()
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1

        # 处理投机解码token
        IF request.spec_token_ids:
            num_scheduled_spec_tokens = (
                num_new_tokens + request.num_computed_tokens - request.num_tokens)
            IF num_scheduled_spec_tokens > 0:
                del request.spec_token_ids[num_scheduled_spec_tokens:]
                scheduled_spec_decode_tokens[request.request_id] = (
                    request.spec_token_ids)

        # 分配编码器cache
        IF encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
            FOR i IN encoder_inputs_to_schedule:
                encoder_cache_manager.allocate(request, i)

    # Step 2: 调度WAITING请求
    IF not preempted_reqs:
        WHILE self.waiting AND token_budget > 0:
            IF len(self.running) == max_num_running_reqs:
                BREAK

            request = self.waiting[0]

            # 检查特殊等待状态
            IF request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # KV传输: 检查远程KV是否已接收
                is_ready = self._update_waiting_for_remote_kv(request)
                IF not is_ready:
                    waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    CONTINUE

            IF request.status == RequestStatus.WAITING_FOR_FSM:
                # 结构化输出: 检查FSM是否已编译
                IF not request.structured_output_request.grammar:
                    waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    CONTINUE

            # 获取已缓存的token
            IF request.num_computed_tokens == 0:
                # 本地缓存
                new_computed_blocks, num_new_local_computed_tokens = (
                    kv_cache_manager.get_computed_blocks(request))

                # 外部缓存 (通过KVConnector)
                IF self.connector:
                    num_external_computed_tokens, load_kv_async = (
                        connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens))

                num_computed_tokens = (
                    num_new_local_computed_tokens + num_external_computed_tokens)
            ELSE:
                # 已有computed tokens (从KV传输恢复)
                new_computed_blocks = kv_cache_manager.create_empty_block_list()
                num_computed_tokens = request.num_computed_tokens

            # KV传输: 异步加载远程KV
            IF load_kv_async:
                num_new_tokens = 0
            ELSE:
                num_new_tokens = request.num_tokens - num_computed_tokens

                # 限制chunk大小
                IF num_new_tokens > long_prefill_token_threshold:
                    num_new_tokens = long_prefill_token_threshold

                num_new_tokens = min(num_new_tokens, token_budget)

            # 调度编码器输入
            IF request.has_encoder_inputs AND not load_kv_async:
                encoder_inputs_to_schedule, num_new_tokens, encoder_budget = (
                    self._try_schedule_encoder_inputs(...))

            IF num_new_tokens == 0 AND not load_kv_async:
                BREAK

            # 分配KV cache
            new_blocks = kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=len(new_computed_blocks.get_all_blocks()),
                new_computed_blocks=new_computed_blocks,
                delay_cache_blocks=load_kv_async,  # KV传输延迟缓存
            )

            IF new_blocks is None:
                BREAK

            # 调度成功
            waiting.popleft()
            running.append(request)
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens

            scheduled_new_reqs.append(request)
            req_to_new_block_ids[request.request_id] = new_blocks.get_block_ids()
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

    # Step 3: 构建SchedulerOutput
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData.from_request(req, req_to_new_block_ids[req.request_id])
            FOR req IN scheduled_new_reqs
        ],
        scheduled_cached_reqs=[
            CachedRequestData.from_request(...)
            FOR req IN (scheduled_resumed_reqs + scheduled_running_reqs)
        ],
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        finished_req_ids=self.finished_req_ids,
        structured_output_request_ids=structured_output_request_ids,
        grammar_bitmask=grammar_bitmask,
    )

    # Step 4: 推进computed_tokens (关键!)
    # 调度完成后立即更新num_computed_tokens
    # 这允许下一步立即重新调度同一请求 (chunked prefill)
    FOR req_id, num_scheduled_token IN num_scheduled_tokens.items():
        self.requests[req_id].num_computed_tokens += num_scheduled_token

    RETURN scheduler_output
```

**关键创新: Step 4的立即更新**
```python
# Line 580-581
# 调度完成后立即更新num_computed_tokens
# 这是V1与V0的重要区别:
# - V0: 在update_from_output后才更新computed tokens
# - V1: 在schedule后立即更新，允许无缝chunked prefill

FOR req_id, num_scheduled_token IN num_scheduled_tokens.items():
    self.requests[req_id].num_computed_tokens += num_scheduled_token
```

#### 1.4 投机解码集成

**update_from_output** 中的投机token处理 (line 700-865):
```python
def update_from_output(
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
    """
    处理模型输出，更新请求状态

    投机解码关键逻辑: 处理拒绝的token
    """

    sampled_token_ids = model_runner_output.sampled_token_ids
    spec_token_ids = model_runner_output.spec_token_ids

    FOR request IN self.running:
        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[req_index]

        # 处理投机解码拒绝
        scheduled_spec_token_ids = (
            scheduler_output.scheduled_spec_decode_tokens.get(req_id))

        IF scheduled_spec_token_ids:
            # 计算被拒绝的token数
            # Example:
            #   scheduled: [A, B, C]  (3个投机token)
            #   generated: [X, A]      (只接受了A)
            #   num_rejected = 3 + 1 - 2 = 2  (拒绝了B, C和bonus)
            num_tokens_rejected = (
                len(scheduled_spec_token_ids) + 1 - len(generated_token_ids))

            # 回退num_computed_tokens (关键!)
            request.num_computed_tokens -= num_tokens_rejected

        # 添加生成的token
        FOR output_token_id IN generated_token_ids:
            request.append_output_token_ids(output_token_id)

            # 检查停止条件
            stopped = check_stop(request, max_model_len)
            IF stopped:
                self._free_request(request)
                BREAK

        # 添加新的投机token (用于下一次调度)
        IF spec_token_ids:
            request.spec_token_ids = spec_token_ids[req_index]

        # 创建输出
        outputs[request.client_index].append(
            EngineCoreOutput(
                request_id=req_id,
                new_token_ids=generated_token_ids,
                finish_reason=request.get_finished_reason(),
                new_logprobs=new_logprobs,
            ))

    RETURN engine_core_outputs
```

---

### 2. V1 KVCacheManager (v1/core/kv_cache_manager.py)

#### 2.1 核心创新点

**增强的KV Cache管理** (line 67-393):
```python
class KVCacheManager:
    """
    V1 KV Cache管理器

    创新点:
    1. KVCacheCoordinator模式: 逻辑与物理分离
    2. 多KV Cache组支持: 不同层可以使用不同cache配置
    3. 增强的前缀缓存: 更智能的block hashing
    4. EAGLE支持: 内置lookahead slots管理
    5. KV事件发布: 支持P/D和offloading
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = False,
        caching_hash_algo: str = "sha256",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ):
        self.kv_cache_config = kv_cache_config
        self.enable_caching = enable_caching
        self.caching_hash_fn = get_hash_fn(caching_hash_algo)

        # 多KV Cache组
        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.num_kv_cache_groups = len(self.kv_cache_groups)

        # BlockPool: 物理块池
        self.block_pool = BlockPool(
            num_blocks_per_group=[
                group.num_gpu_blocks for group in kv_cache_groups
            ],
            block_size_per_group=[
                group.block_size for group in kv_cache_groups
            ],
            enable_caching=enable_caching,
        )

        # KVCacheCoordinator: 逻辑块管理
        self.coordinator = KVCacheCoordinator(
            block_pool=self.block_pool,
            max_model_len=max_model_len,
        )

        # 请求的block hashes (用于前缀缓存)
        self.req_to_block_hashes: dict[str, list[BlockHash]] = {}

        # KV事件发布器 (用于P/D和offloading)
        self.kv_event_publisher = (
            EventPublisher(...) if enable_kv_cache_events else None)
```

#### 2.2 前缀缓存算法

**get_computed_blocks** (line 133-180):
```python
def get_computed_blocks(
    request: Request
) -> tuple[KVCacheBlocks, int]:
    """
    获取请求的已缓存块 (前缀缓存)

    返回:
        (computed_blocks, num_computed_tokens)

    算法:
    1. 计算请求token的block hashes
    2. 查找最长缓存命中
    3. 返回缓存的块和token数
    """

    # 跳过条件
    IF not self.enable_caching OR request.sampling_params.prompt_logprobs:
        RETURN empty_blocks, 0

    # Step 1: 计算block hashes
    block_hashes = hash_request_tokens(
        hash_fn=self.caching_hash_fn,
        block_size=self.block_size,
        request=request,
    )
    # 返回: [hash(tokens[0:16]), hash(tokens[16:32]), ...]

    self.req_to_block_hashes[request.request_id] = block_hashes

    # Step 2: 查找最长缓存命中
    # 注意: 最后一个block必须重新计算 (用于生成新token)
    max_cache_hit_length = request.num_tokens - 1

    computed_blocks, num_new_computed_tokens = (
        self.coordinator.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_cache_hit_length=max_cache_hit_length,
        ))
    # find_longest_cache_hit算法:
    # FOR i, block_hash IN enumerate(block_hashes):
    #     IF block_hash IN block_pool.cached_blocks:
    #         computed_blocks.append(block_pool.get_block(block_hash))
    #     ELSE:
    #         BREAK
    # RETURN computed_blocks, len(computed_blocks) * block_size

    RETURN KVCacheBlocks(computed_blocks), num_new_computed_tokens
```

#### 2.3 块分配算法

**allocate_slots** (line 182-291):
```python
def allocate_slots(
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: Optional[KVCacheBlocks] = None,
    num_draft_tokens: int = 0,          # 投机解码
    num_lookahead_tokens: int = 0,      # EAGLE
    delay_cache_blocks: bool = False,   # KV传输延迟缓存
) -> Optional[KVCacheBlocks]:
    """
    为请求分配KV cache slots

    Blocks布局:
    -----------------------------------------------------------------------
    | < computed > | < new computed > |    < new >    | < pre-allocated > |
    -----------------------------------------------------------------------
    |<------- 已缓存 -------->|<-- 新计算 -->|<--- lookahead/draft --->|

    参数:
        num_new_tokens: 新调度的token数
        num_new_computed_tokens: 新发现的已计算token数 (前缀缓存)
        new_computed_blocks: 新发现的已计算块
        num_draft_tokens: 投机draft token数 (不缓存)
        num_lookahead_tokens: EAGLE lookahead token数
        delay_cache_blocks: 延迟缓存 (用于KV传输)

    返回:
        新分配的块，如果内存不足返回None
    """

    # Step 1: 移除滑动窗口外的块
    self.coordinator.remove_skipped_blocks(
        request.request_id,
        request.num_computed_tokens)

    # Step 2: 计算需要slot的token数
    num_computed_tokens = (
        request.num_computed_tokens + num_new_computed_tokens)
    num_tokens_need_slot = min(
        num_computed_tokens + num_new_tokens + num_lookahead_tokens,
        self.max_model_len)

    # Step 3: 计算需要分配的块数
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens_need_slot=num_tokens_need_slot,
        new_computed_blocks=new_computed_blocks,
    )
    # get_num_blocks_to_allocate算法:
    # existing_blocks = coordinator.get_num_blocks(request_id)
    # total_blocks_needed = ceil(num_tokens_need_slot / block_size)
    # new_blocks_needed = (
    #     total_blocks_needed
    #     - existing_blocks
    #     - len(new_computed_blocks))
    # RETURN new_blocks_needed

    # Step 4: 检查内存是否足够
    IF num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        RETURN None  # 内存不足

    # Step 5: Touch已计算的块 (LRU)
    IF self.enable_caching:
        self.block_pool.touch(new_computed_blocks)

    # Step 6: 保存新计算的块到coordinator
    self.coordinator.save_new_computed_blocks(
        request_id=request.request_id,
        new_computed_blocks=new_computed_blocks,
    )

    # Step 7: 分配新块
    new_blocks = self.coordinator.allocate_new_blocks(
        request_id=request.request_id,
        num_tokens_need_slot=num_tokens_need_slot,
    )
    # allocate_new_blocks算法:
    # num_new_blocks = num_blocks_to_allocate
    # new_blocks = []
    # FOR _ IN range(num_new_blocks):
    #     FOR group IN kv_cache_groups:
    #         block = block_pool.allocate_block(group_id)
    #         new_blocks[group_id].append(block)
    # coordinator.add_blocks(request_id, new_blocks)
    # RETURN new_blocks

    # Step 8: 缓存块 (不包括draft tokens)
    IF self.enable_caching AND not delay_cache_blocks:
        block_hashes = self.req_to_block_hashes[request.request_id]
        num_tokens_to_cache = (
            num_computed_tokens + num_new_tokens - num_draft_tokens)

        self.coordinator.cache_blocks(
            request=request,
            block_hashes=block_hashes,
            num_tokens=num_tokens_to_cache,
        )
        # cache_blocks算法:
        # num_blocks_to_cache = ceil(num_tokens / block_size)
        # blocks = coordinator.get_blocks(request_id)[:num_blocks_to_cache]
        # FOR i, block IN enumerate(blocks):
        #     block_pool.cache_block(block, block_hashes[i])

    # Step 9: 发布KV事件 (用于P/D和offloading)
    IF self.kv_event_publisher:
        self.kv_event_publisher.publish_kv_event(
            request_id=request.request_id,
            event_type=KVEventType.ALLOCATED,
            blocks=new_blocks,
        )

    RETURN KVCacheBlocks(new_blocks)
```

#### 2.4 多KV Cache组支持

**KVCacheGroup**:
```python
# v1/kv_cache_interface.py
@dataclass
class KVCacheGroup:
    """
    KV Cache组定义

    用途:
    - 不同的attention层可以使用不同的cache配置
    - 例如:
      * 前几层使用大块 (128 tokens/block)
      * 后几层使用小块 (16 tokens/block)
    - 或者:
      * MLA (Multi-head Latent Attention) 层使用不同配置
    """

    layer_names: list[str]              # 使用此组的层名列表
    num_gpu_blocks: int                 # GPU块数
    block_size: int                     # 块大小
    num_kv_heads: int                   # KV head数
    head_size: int                      # Head大小
    dtype: torch.dtype                  # 数据类型

# KVCacheConfig
@dataclass
class KVCacheConfig:
    kv_cache_groups: list[KVCacheGroup]

    # Example: Qwen2模型
    # kv_cache_groups = [
    #     KVCacheGroup(
    #         layer_names=[f"model.layers.{i}.self_attn" for i in range(32)],
    #         num_gpu_blocks=1000,
    #         block_size=16,
    #         num_kv_heads=4,  # GQA: 32 query heads, 4 kv heads
    #         head_size=128,
    #         dtype=torch.float16,
    #     )
    # ]
```

---

### 3. V1 Worker (v1/worker/gpu_worker.py)

#### 3.1 核心职责
与V0类似，但有以下增强:
- **KV Cache Spec获取**: 支持多KV cache组
- **编译/Warm up**: 支持CUDA Graph和编译优化
- **Sleep模式**: 支持模型权重offload

#### 3.2 执行流程

**execute_model** (line 283-303):
```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
) -> Optional[ModelRunnerOutput]:
    """
    V1 Worker执行流程

    简化版: 不直接处理KV cache操作
    """

    # Pipeline并行: 接收中间张量
    intermediate_tensors = None
    IF not get_pp_group().is_first_rank():
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group()))

    # 执行模型 (委托给model_runner)
    output = self.model_runner.execute_model(
        scheduler_output,
        intermediate_tensors)

    # Pipeline并行: 发送中间张量
    IF not get_pp_group().is_last_rank():
        ASSERT isinstance(output, IntermediateTensors)
        get_pp_group().send_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group())
        RETURN None

    ASSERT isinstance(output, ModelRunnerOutput)
    RETURN output IF self.is_driver_worker ELSE None
```

**compile_or_warm_up_model** (line 247-277):
```python
def compile_or_warm_up_model() -> None:
    """
    编译或warm up模型

    V1新增: 支持CUDA Graph和torch.compile
    """

    # Warm up sizes (不在CUDA graph capture sizes中的)
    warmup_sizes = vllm_config.compilation_config.compile_sizes.copy()
    IF not model_config.enforce_eager:
        warmup_sizes = [
            x FOR x IN warmup_sizes
            IF x NOT IN cudagraph_capture_sizes
        ]

    FOR size IN sorted(warmup_sizes, reverse=True):
        logger.info("Compile and warming up model for size %d", size)
        self.model_runner._dummy_run(size)

    # CUDA Graph capture
    IF not model_config.enforce_eager:
        self.model_runner.capture_model()

    # Sampler warm up
    IF get_pp_group().is_last_rank():
        max_num_reqs = min(
            scheduler_config.max_num_seqs,
            scheduler_config.max_num_batched_tokens)
        self.model_runner._dummy_sampler_run(
            hidden_states=self.model_runner._dummy_run(max_num_reqs))

    # 重置随机种子 (避免profiling影响推理)
    set_random_seed(model_config.seed)
```

---

### 4. V1 GPUModelRunner (v1/worker/gpu_model_runner.py)

#### 4.1 核心创新点

**execute_model** (line 1171-1508) - V1最复杂的方法:

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, IntermediateTensors]:
    """
    V1模型执行

    创新点:
    1. 状态缓存: 缓存请求状态避免重复构建
    2. 多KV cache组: 每组独立的attention metadata
    3. 编码器分离: 独立执行和缓存多模态编码器
    4. 投机解码集成: 内置drafter和rejection sampler
    5. 结构化输出: Grammar bitmask约束
    6. CUDA Graph: 支持full_cuda_graph模式
    """

    # Step 1: 更新状态
    self._update_states(scheduler_output)
    # 更新 self.input_batch 和 self.requests cache
    # - 添加新请求的状态
    # - 更新running请求的token
    # - 移除finished请求

    # Step 2: 准备输入
    attn_metadata, logits_indices, spec_decode_metadata = (
        self._prepare_inputs(scheduler_output))
    # 返回:
    # - attn_metadata: dict[layer_name, AttentionMetadata]
    #   * V1支持多个KV cache组，每组有独立的metadata
    # - logits_indices: 需要采样的位置索引
    # - spec_decode_metadata: 投机解码元数据 (如果启用)

    # Step 3: 计算token数和padding
    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

    # CUDA Graph: padding到预定义大小
    IF self.use_cuda_graph AND num_scheduled_tokens <= max_cudagraph_size:
        num_input_tokens = vllm_config.pad_for_cudagraph(num_scheduled_tokens)
    ELSE:
        # Eager模式: padding到tp_size倍数 (sequence parallelism)
        IF enable_sequence_parallelism AND tp_size > 1:
            num_input_tokens = round_up(num_scheduled_tokens, tp_size)
        ELSE:
            num_input_tokens = num_scheduled_tokens

    # Data parallelism padding
    num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
    num_input_tokens += num_pad

    # Step 4: 处理多模态输入
    IF self.is_multimodal_model:
        # 执行多模态编码器 (vision encoder)
        self._execute_mm_encoder(scheduler_output)
        # 内部流程:
        # FOR req_id, encoder_input_ids IN scheduled_encoder_inputs.items():
        #     FOR input_id IN encoder_input_ids:
        #         mm_input = request.mm_inputs[input_id]
        #         encoder_output = model.get_multimodal_embeddings(**mm_input)
        #         self.encoder_cache[req_id][input_id] = encoder_output

        # 收集编码器输出
        mm_embeds = self._gather_mm_embeddings(scheduler_output)
        # 返回: dict[req_id, dict[mm_position, embedding]]

    # Step 5: 准备模型输入
    IF self.is_multimodal_model AND get_pp_group().is_first_rank():
        # 多模态模型: 使用embeddings
        input_ids = self.input_ids[:num_scheduled_tokens]
        IF mm_embeds:
            inputs_embeds = self.model.get_input_embeddings(input_ids, mm_embeds)
        ELSE:
            inputs_embeds = self.model.get_input_embeddings(input_ids)

        self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
        inputs_embeds = self.inputs_embeds[:num_input_tokens]
        input_ids = None
    ELSE:
        # 文本模型: 使用token IDs
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None

    # 准备position IDs
    IF self.uses_mrope:  # M-RoPE (Qwen2-VL)
        positions = self.mrope_positions[:, :num_input_tokens]
    ELSE:
        positions = self.positions[:num_input_tokens]

    # Step 6: 执行模型forward
    with set_forward_context(attn_metadata, vllm_config, ...):
        # 设置KV connector (如果启用)
        self.maybe_setup_kv_connector(scheduler_output)

        model_output = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        # 返回: hidden_states 或 (hidden_states, aux_hidden_states)

        # 等待KV保存完成 (如果启用KV传输)
        self.maybe_wait_for_kv_save()

    # Pipeline并行: 发送中间张量
    IF not get_pp_group().is_last_rank():
        RETURN IntermediateTensors({
            "hidden_states": hidden_states,
            ...
        })

    # Step 7: 计算logits
    sample_hidden_states = hidden_states[logits_indices]
    logits = self.model.compute_logits(sample_hidden_states, None)
    # logits: [num_sampled_positions, vocab_size]

    # Step 8: 应用结构化输出约束
    IF scheduler_output.grammar_bitmask is not None:
        self.apply_grammar_bitmask(scheduler_output, logits)
        # 修改logits:
        # FOR req_id, req_index IN structured_output_request_ids.items():
        #     bitmask = grammar_bitmask[req_index]
        #     logits[req_index][bitmask == 0] = -inf

    # Step 9: 采样
    sampling_metadata = self.input_batch.sampling_metadata

    IF spec_decode_metadata is None:
        # 标准采样
        sampler_output = self.sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
    ELSE:
        # 投机解码采样
        # Step 9a: 采样bonus token
        bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
        sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=sampling_metadata,
        )
        bonus_token_ids = sampler_output.sampled_token_ids

        # Step 9b: 拒绝采样
        target_logits = logits[spec_decode_metadata.target_logits_indices]
        output_token_ids = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs (EAGLE不需要)
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )
        # rejection_sampler算法:
        # FOR i, spec_tokens IN enumerate(spec_decode_metadata.draft_token_ids):
        #     target_probs = softmax(target_logits[i])
        #     FOR j, spec_token IN enumerate(spec_tokens):
        #         target_prob = target_probs[spec_token]
        #         draft_prob = 1.0  # EAGLE假设draft prob为1
        #         accept_prob = min(1.0, target_prob / draft_prob)
        #         IF random() > accept_prob:
        #             # 拒绝: 保留到此为止的token + bonus token
        #             accepted = spec_tokens[:j] + [bonus_token_ids[i]]
        #             BREAK
        #     ELSE:
        #         # 全部接受
        #         accepted = spec_tokens + [bonus_token_ids[i]]
        #     output_token_ids[i] = accepted

        sampler_output.sampled_token_ids = output_token_ids

    # Step 10: 处理部分prefill
    # 对于chunked prefill，部分请求不应采样
    discard_sampled_tokens_req_indices = []
    FOR i, req_id IN enumerate(self.input_batch.req_ids):
        req_state = self.requests[req_id]
        seq_len = (
            req_state.num_computed_tokens +
            scheduler_output.num_scheduled_tokens[req_id])

        IF seq_len < req_state.num_tokens:
            # 部分prefill，忽略采样结果
            generator = self.input_batch.generators.get(i)
            IF generator:
                generator.set_offset(generator.get_offset() - 4)  # 回退
            discard_sampled_tokens_req_indices.append(i)

    # Step 11: 计算prompt logprobs (如果需要)
    prompt_logprobs_dict = self._get_prompt_logprobs_dict(
        hidden_states[:num_scheduled_tokens],
        scheduler_output,
    )

    # Step 12: 生成投机token
    spec_token_ids = None
    IF self.speculative_config:
        IF self.speculative_config.method == "ngram":
            spec_token_ids = self.generate_draft_token_ids(...)
        ELIF self.speculative_config.method == "medusa":
            spec_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                ...)
        ELIF self.speculative_config.use_eagle():
            # EAGLE: 使用target model生成draft
            spec_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,  # 刚采样的token
                ...)
            # EAGLE算法:
            # 1. 使用target token embeddings作为输入
            # 2. 通过EAGLE draft model生成多个候选token
            # 3. 返回top-k候选作为spec_token_ids

    # Step 13: 返回输出
    RETURN ModelRunnerOutput(
        req_ids=self.input_batch.req_ids,
        req_id_to_index=self.input_batch.req_id_to_index,
        sampled_token_ids=valid_sampled_token_ids,
        spec_token_ids=spec_token_ids,
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        finished_sending=finished_sending,      # KV传输
        finished_recving=finished_recving,      # KV传输
    )
```

---

## 核心原理对比

### 1. 调度算法对比

| 特性 | V0 | V1 |
|------|-----|-----|
| **调度粒度** | 请求级 (prefill/decode分离) | Token级 (统一调度) |
| **Chunked Prefill** | 支持但需要特殊模式 | 无缝集成 |
| **队列数量** | 3个 (waiting, running, swapped) | 2个 (waiting, running) |
| **computed_tokens更新时机** | update_from_output后 | schedule后立即 |
| **投机解码** | 不支持 | 内置支持 |
| **前缀缓存** | 基础支持 | 增强支持 |
| **状态** | WAITING, RUNNING, SWAPPED | WAITING, WAITING_FOR_FSM, WAITING_FOR_REMOTE_KVS, RUNNING, PREEMPTED |

**调度算法伪代码对比**:

V0默认调度:
```
1. 调度PREFILL请求 (waiting队列)
2. IF 无prefill THEN 调度DECODE请求 (running队列)
3. IF 无抢占 THEN 调度SWAP-IN (swapped队列)
```

V1统一调度:
```
1. 调度RUNNING请求 (可能是prefill或decode)
2. IF 无抢占 THEN 调度WAITING请求 (新请求或恢复的请求)
3. 立即更新num_computed_tokens
```

### 2. 内存管理对比

| 特性 | V0 BlockSpaceManager | V1 KVCacheManager |
|------|----------------------|-------------------|
| **架构模式** | 单体设计 | Coordinator模式 |
| **块池** | CpuGpuBlockAllocator | BlockPool |
| **块表** | BlockTable | KVCacheCoordinator |
| **多组支持** | 不支持 | 支持多KV cache组 |
| **前缀缓存** | Block级哈希 | 增强的Block哈希 + LRU |
| **投机解码** | 不支持 | Lookahead slots支持 |

**块分配算法对比**:

V0 allocate:
```python
block_table = BlockTable(block_size, block_allocator)
block_table.allocate(token_ids=seq.get_token_ids())
self.block_tables[seq.seq_id] = block_table
```

V1 allocate_slots:
```python
# 1. 获取已缓存块
computed_blocks, num_cached = get_computed_blocks(request)

# 2. 计算需要分配的块数
num_blocks_needed = coordinator.get_num_blocks_to_allocate(
    request_id, num_tokens_need_slot, computed_blocks)

# 3. Touch已缓存块 (LRU)
block_pool.touch(computed_blocks)

# 4. 分配新块
new_blocks = coordinator.allocate_new_blocks(request_id, num_tokens_need_slot)

# 5. 缓存块
coordinator.cache_blocks(request, block_hashes, num_tokens)
```

### 3. 模型执行对比

| 特性 | V0 ModelRunner | V1 GPUModelRunner |
|------|----------------|-------------------|
| **状态管理** | 每次重新构建 | 缓存请求状态 |
| **编码器** | 临时处理 | EncoderCacheManager |
| **投机解码** | 不支持 | 内置drafter + rejection sampler |
| **结构化输出** | 不支持 | Grammar bitmask |
| **CUDA Graph** | Piecewise | Piecewise + Full (FA3) |
| **多KV组** | 不支持 | 每组独立metadata |

---

## 以Qwen2模型为例的推理流程

### Qwen2-7B模型配置
```python
model_config = ModelConfig(
    model="Qwen/Qwen2-7B",
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=4,  # GQA: 32 query heads, 4 kv heads
    intermediate_size=14336,  # FFN
    vocab_size=151936,
    max_model_len=32768,  # 32K context
    dtype="auto",  # bfloat16 or float16
)

cache_config = CacheConfig(
    block_size=16,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True,
)
```

### V0推理流程 (Qwen2-7B)

**用户请求**:
```python
prompts = ["请介绍一下vLLM推理引擎"]
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

llm = LLM(model="Qwen/Qwen2-7B")
outputs = llm.generate(prompts, sampling_params)
```

**内部流程**:

1. **Tokenization & 添加请求**:
```python
# Tokenize
prompt_token_ids = tokenizer.encode("请介绍一下vLLM推理引擎")
# [1, 108, 100, 2500, 3221, 20145, 106, 110, 118, 101, 108, 76, 52, 43,
#  41, 120956, 122337]  # 17 tokens

# 创建Sequence
seq = Sequence(
    seq_id=0,
    prompt_token_ids=prompt_token_ids,
    block_size=16,
    eos_token_id=151643,
)
seq.data = SequenceData(
    prompt_token_ids=prompt_token_ids,
    output_token_ids=[],
    cumulative_logprob=0.0,
)

# 创建SequenceGroup
seq_group = SequenceGroup(
    request_id="0",
    seqs=[seq],
    sampling_params=sampling_params,
    arrival_time=time.time(),
)

# 添加到Scheduler
scheduler.add_seq_group(seq_group)  # 添加到waiting队列
```

2. **第1次step: Prefill**:
```python
# Schedule
scheduler_outputs = scheduler.schedule()

# Prefill调度:
# - 计算需要的块数: ceil(17 / 16) = 2 blocks
# - 检查GPU内存: can_allocate() -> OK
# - 分配块: block_table = [0, 1]  (物理块ID)
# - 移动到running队列

# SchedulerOutputs:
scheduled_seq_groups = [
    ScheduledSequenceGroup(
        seq_group=seq_group_0,
        token_chunk_size=17,  # 全部prefill
    )
]
num_batched_tokens = 17
num_prefill_groups = 1

# 构建ExecuteModelRequest
execute_model_req = ExecuteModelRequest(
    seq_group_metadata_list=[
        SequenceGroupMetadata(
            request_id="0",
            is_prompt=True,
            seq_data={0: seq.data},
            block_tables={0: [0, 1]},
            token_chunk_size=17,
        )
    ],
)

# Worker执行
model_runner.prepare_model_input():
    input_ids = tensor([1, 108, 100, ..., 122337])  # [17]
    positions = tensor([0, 1, 2, ..., 16])          # [17]
    slot_mapping = tensor([0,1,2,...,15,16])        # 块0: 0-15, 块1: 16

model.forward():
    # Qwen2ForCausalLM
    hidden_states = embed_tokens(input_ids)  # [17, 4096]

    FOR layer IN range(32):
        # Self-Attention with GQA
        hidden_states = layer.self_attn(
            hidden_states,
            positions=positions,
            kv_cache=kv_caches[layer],  # (K[num_blocks, 16, 4, 128], V[...])
            attn_metadata=attn_metadata,
        )
        # 内部: 写入KV到slot_mapping指定的位置

        # FFN
        hidden_states = layer.mlp(hidden_states)  # [17, 4096]

    hidden_states = norm(hidden_states)  # [17, 4096]

logits = lm_head(hidden_states)  # [17, 151936]

# 采样: 只对最后一个token
last_token_logits = logits[-1:]  # [1, 151936]
sampled_token_id = sampler(last_token_logits)  # e.g., 220 ("v")

# SamplerOutput:
sampler_output = SamplerOutput(
    outputs=[
        CompletionSequenceGroupOutput(
            samples=[
                SequenceOutput(
                    output_token=220,
                    logprobs={220: -0.01, ...},
                )
            ]
        )
    ]
)

# 更新序列
seq.append_token_id(220)
# seq.data.output_token_ids = [220]
seq.data.update_num_computed_tokens(17)
```

3. **第2次step: Decode**:
```python
# Schedule
# 现在seq.get_len() = 18 (17 prompt + 1 output)
# num_computed_tokens = 17
# num_uncomputed_tokens = 1

scheduler_outputs = scheduler.schedule()

# Decode调度:
# - 当前块: [0, 1], 块1已有17个token (1个空位)
# - 需要1个新slot (位置18需要新块)
# - 分配新块: block_table = [0, 1, 2]

# SchedulerOutputs:
scheduled_seq_groups = [
    ScheduledSequenceGroup(
        seq_group=seq_group_0,
        token_chunk_size=1,  # decode每次1个
    )
]
num_batched_tokens = 1
num_prefill_groups = 0

# Worker执行
model_runner.prepare_model_input():
    input_ids = tensor([220])      # 上次生成的token
    positions = tensor([17])        # position = 17
    slot_mapping = tensor([32])     # 块2的第0个位置 (2*16+0=32)

model.forward():
    hidden_states = embed_tokens(input_ids)  # [1, 4096]

    FOR layer IN range(32):
        hidden_states = layer.self_attn(
            hidden_states,
            positions=positions,
            kv_cache=kv_caches[layer],
            attn_metadata=attn_metadata,
        )
        # Attention需要访问所有18个token的KV
        # seq_lens = [18]
        # block_tables = [[0, 1, 2]]

        hidden_states = layer.mlp(hidden_states)

    hidden_states = norm(hidden_states)

logits = lm_head(hidden_states)  # [1, 151936]

sampled_token_id = sampler(logits)  # e.g., 76 ("L")

# 更新序列
seq.append_token_id(76)
# seq.data.output_token_ids = [220, 76]  ("vL")
seq.data.update_num_computed_tokens(18)

# 重复decode直到:
# - 生成EOS token (151643)
# - 达到max_tokens (100)
# - 其他停止条件
```

### V1推理流程 (Qwen2-7B)

**用户请求** (相同):
```python
prompts = ["请介绍一下vLLM推理引擎"]
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

llm = LLM(model="Qwen/Qwen2-7B", enforce_eager=False)
outputs = llm.generate(prompts, sampling_params)
```

**内部流程**:

1. **Preprocessing & 添加请求**:
```python
# Processor.preprocess
preprocessed = processor.preprocess(
    prompt="请介绍一下vLLM推理引擎",
    request_id="0",
    tokenizer=tokenizer,
)
# EngineCoreRequest:
engine_core_request = EngineCoreRequest(
    request_id="0",
    prompt_token_ids=[1, 108, 100, 2500, ...],  # 17 tokens
    mm_inputs=None,
    mm_hashes=None,
    mm_placeholders=None,
    sampling_params=sampling_params,
    eos_token_id=151643,
    arrival_time=time.time(),
)

# EngineCore.add_request
request = Request.from_engine_core_request(engine_core_request)
# Request:
request = Request(
    request_id="0",
    prompt_token_ids=[1, 108, 100, ...],
    _output_token_ids=[],
    _all_token_ids=[1, 108, 100, ...],  # 17 tokens
    spec_token_ids=[],
    num_computed_tokens=0,
    status=RequestStatus.WAITING,
    sampling_params=sampling_params,
)

# Scheduler.add_request
scheduler.add_request(request)  # 添加到waiting队列
```

2. **第1次step: Unified Prefill**:
```python
# Schedule
scheduler_output = scheduler.schedule()

# 统一调度逻辑:
# 1. 调度RUNNING请求: (空)
# 2. 调度WAITING请求:
#    - 计算num_new_tokens = 17 - 0 = 17
#    - 前缀缓存检查:
#      computed_blocks, num_cached = kv_cache_manager.get_computed_blocks(request)
#      # 假设无缓存命中: computed_blocks=[], num_cached=0
#    - 分配KV cache:
#      new_blocks = kv_cache_manager.allocate_slots(
#          request, num_new_tokens=17, num_new_computed_tokens=0)
#      # 分配2个块: [[0, 1]]  (单KV cache组)
#    - 移动到running队列
#    - 立即更新: request.num_computed_tokens = 0 + 17 = 17

# SchedulerOutput:
scheduled_new_reqs = [
    NewRequestData(
        req_id="0",
        prompt_token_ids=[1, 108, 100, ...],
        block_ids=([0, 1],),  # tuple of lists
        num_computed_tokens=0,  # 调度前的值
    )
]
scheduled_cached_reqs = []
num_scheduled_tokens = {"0": 17}
total_num_scheduled_tokens = 17

# Worker执行
model_runner.execute_model(scheduler_output):
    # Step 1: 更新状态
    _update_states(scheduler_output):
        # 添加新请求到self.requests cache
        self.requests["0"] = CachedRequestState(
            request_id="0",
            prompt_token_ids=[1, 108, 100, ...],
            output_token_ids=[],
            num_computed_tokens=0,
            block_ids=([0, 1],),
        )
        # 构建input_batch
        self.input_batch.req_ids = ["0"]
        self.input_batch.req_id_to_index = {"0": 0}

    # Step 2: 准备输入
    attn_metadata, logits_indices, spec_decode_metadata = _prepare_inputs(...)
    # attn_metadata: dict[layer_name, FlashAttentionMetadata]
    # V1为每个KV cache组构建独立的metadata
    # 对于Qwen2 (单组):
    attn_metadata = {
        "model.layers.0.self_attn": FlashAttentionMetadata(
            query_start_loc=tensor([0, 17]),  # [num_reqs+1]
            seq_lens=tensor([17]),             # [num_reqs]
            max_query_len=17,
            max_kv_len=17,
            slot_mapping=tensor([0,1,...,16]), # [17]
            block_table=tensor([[0, 1]]),      # [num_reqs, max_blocks]
        ),
    }
    logits_indices = tensor([16])  # 只对最后一个token采样

    # Step 3: 模型forward
    input_ids = self.input_ids[:17]  # [1, 108, 100, ...]
    positions = self.positions[:17]   # [0, 1, 2, ..., 16]

    model_output = self.model(
        input_ids=input_ids,
        positions=positions,
    )
    # Qwen2 forward (与V0类似但有优化):
    hidden_states = embed_tokens(input_ids)  # [17, 4096]

    FOR layer IN range(32):
        # 使用FlashAttention-2
        hidden_states = layer.self_attn(
            hidden_states,
            positions=positions,
            kv_cache=kv_caches[layer],
            attn_metadata=attn_metadata[layer.name],
        )
        # GQA: 32 query heads, 4 kv heads
        # FlashAttention kernel直接写入KV cache

        hidden_states = layer.mlp(hidden_states)

    hidden_states = norm(hidden_states)

    # Step 4: 计算logits
    sample_hidden_states = hidden_states[logits_indices]  # [1, 4096]
    logits = model.compute_logits(sample_hidden_states)   # [1, 151936]

    # Step 5: 采样
    sampler_output = self.sampler(logits, sampling_metadata)
    sampled_token_id = sampler_output.sampled_token_ids  # [[220]]

    # Step 6: 生成投机token (如果启用)
    IF self.speculative_config:
        # 假设使用EAGLE, num_spec_tokens=3
        spec_token_ids = self.drafter.propose(
            target_token_ids=[220],
            target_positions=[16],
            target_hidden_states=hidden_states[-1:],
            next_token_ids=[220],
        )
        # 返回: [[76, 76, 12345]]  # 3个draft tokens

    # ModelRunnerOutput:
    RETURN ModelRunnerOutput(
        req_ids=["0"],
        req_id_to_index={"0": 0},
        sampled_token_ids=[[220]],
        spec_token_ids=[[76, 76, 12345]] IF speculative_config ELSE None,
        logprobs=[...],
    )

# Scheduler.update_from_output
update_from_output(scheduler_output, model_runner_output):
    request = self.requests["0"]

    # 添加生成的token
    request.append_output_token_ids(220)
    # request._output_token_ids = [220]
    # request._all_token_ids = [1, 108, ..., 122337, 220]

    # 添加投机token
    IF model_runner_output.spec_token_ids:
        request.spec_token_ids = [76, 76, 12345]
        # request.num_tokens_with_spec = 18 + 3 = 21

    # 检查停止条件
    stopped = check_stop(request, max_model_len)
    IF not stopped:
        new_running.append(request)

    # 创建输出
    RETURN EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id="0",
                new_token_ids=[220],
                finish_reason=None,
                new_logprobs=[...],
            )
        ],
    )
```

3. **第2次step: Decode + Spec Verify**:
```python
# Schedule
# 当前状态:
# - request.num_tokens = 18 (17 prompt + 1 output)
# - request.spec_token_ids = [76, 76, 12345]
# - request.num_tokens_with_spec = 21
# - request.num_computed_tokens = 17 (上次schedule后更新)

scheduler_output = scheduler.schedule()

# 统一调度逻辑:
# 1. 调度RUNNING请求:
#    - num_new_tokens = 21 - 17 = 4  (1 real + 3 spec)
#    - 分配KV cache (包括lookahead):
#      new_blocks = kv_cache_manager.allocate_slots(
#          request,
#          num_new_tokens=4,
#          num_draft_tokens=3,  # 不缓存
#          num_lookahead_tokens=3)  # EAGLE lookahead
#      # 可能需要新块: [[0, 1, 2]]
#    - num_scheduled_tokens["0"] = 4
#    - scheduled_spec_decode_tokens["0"] = [76, 76, 12345]
#    - 立即更新: request.num_computed_tokens = 17 + 4 = 21

# Worker执行
model_runner.execute_model(scheduler_output):
    # Step 1: 更新状态
    self.requests["0"].num_computed_tokens = 17
    self.requests["0"].output_token_ids = [220]
    self.requests["0"].block_ids = ([0, 1, 2],)

    # 构建input: 1 real token + 3 spec tokens
    input_batch.input_tokens = [220, 76, 76, 12345]

    # Step 2: 准备输入 (投机解码模式)
    attn_metadata, logits_indices, spec_decode_metadata = _prepare_inputs(...)
    # spec_decode_metadata:
    spec_decode_metadata = SpecDecodeMetadata(
        draft_token_ids=tensor([76, 76, 12345]),  # 3个投机token
        num_draft_tokens=[3],
        cu_num_draft_tokens=tensor([0, 3]),
        target_logits_indices=tensor([0, 1, 2]),  # 验证3个draft
        bonus_logits_indices=tensor([3]),         # 生成bonus token
        logits_indices=tensor([0, 1, 2, 3]),
    )

    # Step 3: 模型forward
    input_ids = [220, 76, 76, 12345]
    positions = [17, 18, 19, 20]

    hidden_states = model(input_ids, positions)  # [4, 4096]

    # Step 4: 计算logits
    logits = model.compute_logits(hidden_states)  # [4, 151936]

    # Step 5: 投机解码采样
    # 5a: 采样bonus token
    bonus_logits = logits[3:]  # [1, 151936]
    bonus_token_id = sampler(bonus_logits)  # e.g., 77

    # 5b: 拒绝采样
    target_logits = logits[:3]  # [3, 151936]
    output_token_ids = rejection_sampler(
        spec_decode_metadata,
        target_logits,
        bonus_token_id=[77],
        sampling_metadata,
    )
    # 假设拒绝第3个token:
    # - 检查token 76 at position 18: ACCEPT (prob > threshold)
    # - 检查token 76 at position 19: ACCEPT
    # - 检查token 12345 at position 20: REJECT
    # output_token_ids = [[76, 76, 77]]  # 2 accepted + bonus

    # Step 6: 生成新的投机token
    spec_token_ids = self.drafter.propose(
        target_token_ids=[220, 76, 76, 77],
        target_positions=[17, 18, 19, 20],
        target_hidden_states=hidden_states,
        next_token_ids=[77],
    )
    # 返回: [[88, 99, 111]]  # 新的3个draft tokens

    # ModelRunnerOutput:
    RETURN ModelRunnerOutput(
        req_ids=["0"],
        req_id_to_index={"0": 0},
        sampled_token_ids=[[76, 76, 77]],  # 3个token被接受
        spec_token_ids=[[88, 99, 111]],
        logprobs=[...],
    )

# Scheduler.update_from_output
update_from_output(scheduler_output, model_runner_output):
    request = self.requests["0"]

    # 处理投机解码拒绝
    # scheduled: [76, 76, 12345] (3个)
    # generated: [76, 76, 77]    (3个)
    # num_rejected = 3 + 1 - 3 = 1  (拒绝了12345)
    request.num_computed_tokens -= 1  # 21 -> 20

    # 添加生成的token
    FOR token_id IN [76, 76, 77]:
        request.append_output_token_ids(token_id)
    # request._output_token_ids = [220, 76, 76, 77]
    # request._all_token_ids = [1, 108, ..., 220, 76, 76, 77]  # 21 tokens

    # 添加新的投机token
    request.spec_token_ids = [88, 99, 111]
    # request.num_tokens_with_spec = 21 + 3 = 24

    # 重复直到完成...
```

---

## 总结

### V0核心原理

1. **分阶段调度**: Prefill和Decode严格分离
2. **请求级管理**: 以SequenceGroup为调度单位
3. **块级内存**: PagedAttention的块表映射
4. **同步执行**: 调度和执行顺序进行

### V1核心原理

1. **统一Token级调度**: 无prefill/decode区分
2. **立即状态更新**: schedule后立即更新num_computed_tokens
3. **Coordinator模式**: 逻辑块管理与物理块池分离
4. **原生投机解码**: 内置drafter和rejection sampler
5. **多模态优化**: 独立的EncoderCacheManager
6. **异步执行**: EngineCoreClient抽象支持多进程

### 性能优势 (V1 vs V0)

| 优化 | V0 | V1 | 收益 |
|------|-----|-----|------|
| Chunked Prefill | 需要特殊模式 | 无缝集成 | 降低TTFT |
| 投机解码 | 不支持 | 内置 | 2-3x生成速度 |
| 前缀缓存 | 基础 | 增强LRU | 更高命中率 |
| 编码器缓存 | 临时 | 专门管理 | 避免重复计算 |
| 调度效率 | 阶段切换 | 无切换开销 | 更高吞吐量 |

---

## 附录: 关键代码位置

| 组件 | V0位置 | V1位置 |
|------|---------|---------|
| Scheduler.schedule | core/scheduler.py:1488-1682 | v1/core/sched/scheduler.py:158-584 |
| Scheduler更新 | engine/llm_engine.py:931-1164 | v1/core/sched/scheduler.py:700-865 |
| BlockManager.allocate | core/block_manager.py:166-206 | - |
| KVCacheManager.allocate | - | v1/core/kv_cache_manager.py:182-291 |
| Worker.execute_model | worker/worker.py:387-449 | v1/worker/gpu_worker.py:283-303 |
| ModelRunner.execute | worker/model_runner.py | v1/worker/gpu_model_runner.py:1171-1508 |
