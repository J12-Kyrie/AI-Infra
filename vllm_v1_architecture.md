# vLLM V1 架构详解 - Qwen3 离线推理流程

## 目录
1. [架构概览](#架构概览)
2. [核心组件](#核心组件)
3. [完整数据流](#完整数据流)
4. [详细执行流程](#详细执行流程)
5. [关键数据类型](#关键数据类型)
6. [V0与V1对比](#v0与v1对比)

---

## 架构概览

vLLM V1 架构是对V0的重大升级，采用**异步/多进程**执行模式，针对高吞吐量和低延迟场景优化。核心设计理念是通过更精细的调度、增强的KV cache管理和内置的投机解码支持来最大化性能。

### 架构层次
```
用户层 (LLM)
    ↓
引擎层 (LLMEngine + EngineCoreClient)
    ↓
核心层 (EngineCore + Scheduler + KVCacheManager)
    ↓
执行层 (Worker → GPUModelRunner)
    ↓
模型层 (Model + Sampler + Drafter)
```

### V1关键创新
1. **EngineCoreClient**: 异步执行抽象，支持多进程/多线程
2. **统一调度**: 无prefill/decode阶段区分，统一Token级调度
3. **增强KV Cache**: KVCacheManager + Coordinator模式
4. **内置投机解码**: 原生支持EAGLE, Medusa, Ngram
5. **多模态优化**: EncoderCacheManager处理视觉编码器
6. **KV传输**: 支持P/D(Prefill/Decode分离)和卸载

---

## 核心组件

### 1. LLM (entrypoints/llm.py)

**职责**: 用户API接口，与V0基本相同

**关键方法**:
```python
def generate(
    prompts: Union[PromptType, Sequence[PromptType]],
    sampling_params: Optional[Union[SamplingParams, Sequence[SamplingParams]]] = None,
) -> list[RequestOutput]
```

与V0相同的接口，内部委托给V1 LLMEngine。

---

### 2. LLMEngine (v1/engine/llm_engine.py:44-318)

**职责**: V1引擎协调器，使用EngineCoreClient异步执行

**关键属性**:
```python
self.model_config: ModelConfig
self.processor: Processor                    # 输入预处理
self.tokenizer: AnyTokenizer
self.detokenizer: Detokenizer
self.output_processor: SequenceGroupOutputProcessor
self.engine_core: EngineCoreClient          # 核心执行客户端
```

**核心方法**:

#### 2.1 add_request (line 116-165)
```python
def add_request(
    request_id: str,
    prompt: PromptType,
    params: Union[SamplingParams, PoolingParams],
    arrival_time: Optional[float] = None,
) -> None:
    # 1. 预处理输入
    preprocessed = self.processor.preprocess(prompt, ...)
    # 返回: EngineCoreRequest

    # 2. 发送到EngineCore
    self.engine_core.add_request(preprocessed)
```

**与V0的区别**:
- 使用`Processor`代替`InputPreprocessor`
- 创建`EngineCoreRequest`而不是`Sequence/SequenceGroup`
- 通过`EngineCoreClient`异步添加

#### 2.2 step (line 224-250)
```python
def step(self) -> list[RequestOutput]:
    # 1. 从EngineCore获取输出
    outputs: EngineCoreOutputs = self.engine_core.get_output()

    # 2. 处理输出
    processed_outputs = self.output_processor.process_outputs(
        outputs.outputs,
        engine_core_timestamp=outputs.timestamp,
    )

    return processed_outputs.request_outputs
```

**与V0的区别**:
- 不直接调度和执行，而是从EngineCoreClient获取结果
- EngineCoreClient内部运行独立的EngineCore进程/线程
- 输出处理流程更简化

---

### 3. Scheduler (v1/core/sched/scheduler.py:38-1045)

**职责**: V1统一调度器，Token级调度，无prefill/decode区分

**关键属性**:
```python
self.requests: dict[str, Request]           # req_id -> Request
self.waiting: deque[Request]                # 等待队列
self.running: list[Request]                 # 运行队列
self.kv_cache_manager: KVCacheManager      # KV cache管理
self.encoder_cache_manager: EncoderCacheManager  # 编码器cache
self.structured_output_manager: StructuredOutputManager
self.connector: Optional[KVConnectorBase_V1]  # KV传输
```

**核心方法**:

#### 3.1 schedule (line 158-584)

**调度算法核心思想** (line 159-168):
```
V1调度器没有"prefill阶段"和"decode阶段"的概念。
每个请求有两个关键属性:
- num_computed_tokens: 已计算的token数
- num_tokens_with_spec: 总需求token数 =
    len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)

调度目标: 让每个请求的 num_computed_tokens 追赶其 num_tokens_with_spec

这种设计统一支持:
- Chunked prefills (分块prefill)
- Prefix caching (前缀缓存)
- Speculative decoding (投机解码)
- Future: Jump decoding
```

**执行流程**:

**步骤1: 调度RUNNING请求** (line 196-305)
```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # 计算需要调度的token数
    num_new_tokens = (request.num_tokens_with_spec -
                      request.num_computed_tokens)
    num_new_tokens = min(num_new_tokens, token_budget)

    # 调度编码器输入 (如果是多模态)
    encoder_inputs_to_schedule = self._try_schedule_encoder_inputs(...)

    # 分配KV cache块
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, ...)

    if new_blocks is None:
        # 内存不足，抢占最低优先级请求
        preempted_req = self.running.pop()
        self.kv_cache_manager.free(preempted_req)
        preempted_req.status = RequestStatus.PREEMPTED
        self.waiting.appendleft(preempted_req)

    # 记录调度信息
    num_scheduled_tokens[request.request_id] = num_new_tokens
    token_budget -= num_new_tokens
```

**步骤2: 调度WAITING请求** (line 318-485)
```python
while self.waiting and token_budget > 0:
    request = self.waiting[0]

    # 检查特殊等待状态
    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        # KV传输等待
        is_ready = self._update_waiting_for_remote_kv(request)
        if not is_ready:
            continue

    if request.status == RequestStatus.WAITING_FOR_FSM:
        # 结构化输出FSM编译等待
        if not request.structured_output_request.grammar:
            continue

    # 前缀缓存: 获取已计算的token
    if request.num_computed_tokens == 0:
        # 本地缓存
        new_computed_blocks, num_new_local_computed_tokens = \
            self.kv_cache_manager.get_computed_blocks(request)

        # 外部缓存 (通过KVConnector)
        if self.connector:
            num_external_computed_tokens, load_kv_async = \
                self.connector.get_num_new_matched_tokens(...)

        num_computed_tokens = (num_new_local_computed_tokens +
                               num_external_computed_tokens)

    # 分配KV cache
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, ...)

    self.waiting.popleft()
    self.running.append(request)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = num_computed_tokens
```

**步骤3: 构建SchedulerOutput** (line 516-556)
```python
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=new_reqs_data,
    scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
    num_scheduled_tokens=num_scheduled_tokens,
    total_num_scheduled_tokens=total_num_scheduled_tokens,
    scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
    scheduled_encoder_inputs=scheduled_encoder_inputs,
    num_common_prefix_blocks=num_common_prefix_blocks,
    finished_req_ids=self.finished_req_ids,
    grammar_bitmask=grammar_bitmask,  # 结构化输出
)
```

**步骤4: 推进computed_tokens** (line 580-581)
```python
# 调度完成后，立即更新num_computed_tokens
# 这允许在下一步立即重新调度同一请求 (chunked prefill)
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    self.requests[req_id].num_computed_tokens += num_scheduled_token
```

#### 3.2 update_from_output (line 700-865)

**职责**: 处理模型输出，更新请求状态

**执行流程**:
```python
def update_from_output(
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:

    sampled_token_ids = model_runner_output.sampled_token_ids
    spec_token_ids = model_runner_output.spec_token_ids
    logprobs = model_runner_output.logprobs

    for request in self.running:
        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[req_index]

        # 投机解码: 处理拒绝的token
        scheduled_spec_token_ids = (
            scheduler_output.scheduled_spec_decode_tokens.get(req_id))
        if scheduled_spec_token_ids:
            num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                   len(generated_token_ids))
            request.num_computed_tokens -= num_tokens_rejected

        # 添加生成的token
        for output_token_id in generated_token_ids:
            request.append_output_token_ids(output_token_id)

            # 检查停止条件
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                self._free_request(request)
                break

        # 添加投机token
        if spec_token_ids:
            request.spec_token_ids = spec_token_ids[req_index]

        # 创建输出
        outputs[request.client_index].append(
            EngineCoreOutput(
                request_id=req_id,
                new_token_ids=generated_token_ids,
                finish_reason=request.get_finished_reason(),
                new_logprobs=new_logprobs,
            ))

    return engine_core_outputs
```

---

### 4. KVCacheManager (v1/core/kv_cache_manager.py:67-393)

**职责**: V1 KV cache管理，替代V0的BlockSpaceManager

**关键属性**:
```python
self.coordinator: KVCacheCoordinator   # 实际管理逻辑
self.block_pool: BlockPool              # 物理块池
self.req_to_block_hashes: dict[str, list[BlockHash]]  # 前缀缓存哈希
```

**核心方法**:

#### 4.1 get_computed_blocks (line 133-180)
```python
def get_computed_blocks(
    request: Request
) -> tuple[KVCacheBlocks, int]:
    """获取请求的已缓存块 (前缀缓存)

    Returns:
        (computed_blocks, num_computed_tokens)
    """
    # 如果禁用缓存或需要prompt logprobs，跳过
    if not self.enable_caching or request.sampling_params.prompt_logprobs:
        return empty_blocks, 0

    # 计算block hashes
    block_hashes = hash_request_tokens(
        self.caching_hash_fn, self.block_size, request)

    # 查找最长缓存命中
    max_cache_hit_length = request.num_tokens - 1  # 必须重计算最后一个token
    computed_blocks, num_new_computed_tokens = \
        self.coordinator.find_longest_cache_hit(
            block_hashes, max_cache_hit_length)

    return KVCacheBlocks(computed_blocks), num_new_computed_tokens
```

#### 4.2 allocate_slots (line 182-291)
```python
def allocate_slots(
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: Optional[KVCacheBlocks] = None,
    num_lookahead_tokens: int = 0,  # EAGLE等投机解码
    delay_cache_blocks: bool = False,  # KV传输延迟缓存
) -> Optional[KVCacheBlocks]:
    """为请求分配KV cache槽位

    Blocks布局:
    -----------------------------------------------------------------------
    | < computed > | < new computed > |    < new >    | < pre-allocated > |
    -----------------------------------------------------------------------

    Returns:
        新分配的块，如果内存不足返回None
    """
    # 释放滑动窗口外的块
    self.coordinator.remove_skipped_blocks(
        request.request_id, request.num_computed_tokens)

    num_computed_tokens = (request.num_computed_tokens +
                           num_new_computed_tokens)
    num_tokens_need_slot = min(
        num_computed_tokens + num_new_tokens + num_lookahead_tokens,
        self.max_model_len)

    # 计算需要分配的块数
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id, num_tokens_need_slot, new_computed_blocks)

    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        return None  # 内存不足

    # Touch已计算的块 (LRU)
    if self.enable_caching:
        self.block_pool.touch(new_computed_blocks)

    # 保存新计算的块
    self.coordinator.save_new_computed_blocks(
        request_id, new_computed_blocks)

    # 分配新块
    new_blocks = self.coordinator.allocate_new_blocks(
        request_id, num_tokens_need_slot)

    # 缓存块 (不包括投机token)
    if self.enable_caching and not delay_cache_blocks:
        self.coordinator.cache_blocks(
            request, block_hashes,
            num_computed_tokens + num_new_tokens - num_draft_tokens)

    return KVCacheBlocks(new_blocks)
```

---

### 5. GPUModelRunner (v1/worker/gpu_model_runner.py:77-2321)

**职责**: V1模型运行器，相比V0大幅增强

**关键属性**:
```python
self.model: nn.Module
self.sampler: Sampler                      # 采样器
self.drafter: Union[NgramProposer, EagleProposer, MedusaProposer]  # 投机解码
self.rejection_sampler: RejectionSampler   # 投机解码验证
self.requests: dict[str, CachedRequestState]  # 缓存的请求状态
self.input_batch: InputBatch               # 持久化批次
self.encoder_cache: dict[str, dict[int, torch.Tensor]]  # 编码器输出缓存
```

**核心方法**:

#### 5.1 execute_model (line 1171-1508)

**职责**: 执行模型推理，V1最复杂的方法

**执行流程**:

**步骤1: 更新状态** (line 1177)
```python
self._update_states(scheduler_output)
# 更新 self.input_batch 和 self.requests
# 添加新请求、更新running请求、移除finished请求
```

**步骤2: 准备输入** (line 1186-1187)
```python
attn_metadata, logits_indices, spec_decode_metadata = \
    self._prepare_inputs(scheduler_output)
# 返回:
# - attn_metadata: dict[layer_name, AttentionMetadata]
# - logits_indices: 需要采样的位置
# - spec_decode_metadata: 投机解码元数据
```

**步骤3: 计算token数和padding** (line 1188-1209)
```python
num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

if self.use_cuda_graph and num_scheduled_tokens <= max_cudagraph_size:
    # CUDA Graph: padding到预定义大小
    num_input_tokens = self.vllm_config.pad_for_cudagraph(
        num_scheduled_tokens)
else:
    # Eager模式: padding到tp_size倍数 (sequence parallelism)
    if enable_sequence_parallelism and tp_size > 1:
        num_input_tokens = round_up(num_scheduled_tokens, tp_size)
    else:
        num_input_tokens = num_scheduled_tokens

# Data parallelism padding
num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
num_input_tokens += num_pad
```

**步骤4: 处理多模态输入** (line 1213-1218)
```python
if self.is_multimodal_model:
    # 运行多模态编码器 (vision encoder)
    self._execute_mm_encoder(scheduler_output)
    # 收集编码器输出
    mm_embeds = self._gather_mm_embeddings(scheduler_output)
```

**步骤5: 准备模型输入** (line 1220-1244)
```python
if self.is_multimodal_model and get_pp_group().is_first_rank:
    # 多模态模型: 使用embeddings
    input_ids = self.input_ids[:num_scheduled_tokens]
    if mm_embeds:
        inputs_embeds = self.model.get_input_embeddings(
            input_ids, mm_embeds)
    else:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
    self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
    inputs_embeds = self.inputs_embeds[:num_input_tokens]
    input_ids = None
else:
    # 文本模型: 使用token IDs
    input_ids = self.input_ids[:num_input_tokens]
    inputs_embeds = None

if self.uses_mrope:  # M-RoPE (Qwen2-VL)
    positions = self.mrope_positions[:, :num_input_tokens]
else:
    positions = self.positions[:num_input_tokens]
```

**步骤6: 执行模型forward** (line 1254-1265)
```python
with set_forward_context(attn_metadata, self.vllm_config, ...):
    self.maybe_setup_kv_connector(scheduler_output)

    model_output = self.model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
    )
    # 返回: hidden_states 或 (hidden_states, aux_hidden_states)

    self.maybe_wait_for_kv_save()
```

**步骤7: 计算logits** (line 1289-1292)
```python
if get_pp_group().is_last_rank:
    sample_hidden_states = hidden_states[logits_indices]
    logits = self.model.compute_logits(sample_hidden_states, None)
```

**步骤8: 应用结构化输出约束** (line 1303-1304)
```python
if scheduler_output.grammar_bitmask is not None:
    self.apply_grammar_bitmask(scheduler_output, logits)
```

**步骤9: 采样** (line 1307-1337)
```python
sampling_metadata = self.input_batch.sampling_metadata

if spec_decode_metadata is None:
    # 标准采样
    sampler_output = self.sampler(
        logits=logits,
        sampling_metadata=sampling_metadata,
    )
else:
    # 投机解码采样
    bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
    sampler_output = self.sampler(
        logits=bonus_logits,
        sampling_metadata=sampling_metadata,
    )
    bonus_token_ids = sampler_output.sampled_token_ids

    # 拒绝采样
    target_logits = logits[spec_decode_metadata.target_logits_indices]
    output_token_ids = self.rejection_sampler(
        spec_decode_metadata,
        None,  # draft_probs
        target_logits,
        bonus_token_ids,
        sampling_metadata,
    )
    sampler_output.sampled_token_ids = output_token_ids
```

**步骤10: 处理部分prefill** (line 1341-1355)
```python
# 对于chunked prefill，部分请求不应采样
discard_sampled_tokens_req_indices = []
for i, req_id in enumerate(self.input_batch.req_ids):
    req_state = self.requests[req_id]
    seq_len = (req_state.num_computed_tokens +
               scheduler_output.num_scheduled_tokens[req_id])
    if seq_len < req_state.num_tokens:
        # 部分prefill，忽略采样结果
        generator = self.input_batch.generators.get(i)
        if generator:
            generator.set_offset(generator.get_offset() - 4)  # 回退
        discard_sampled_tokens_req_indices.append(i)
```

**步骤11: 计算prompt logprobs** (line 1364-1367)
```python
prompt_logprobs_dict = self._get_prompt_logprobs_dict(
    hidden_states[:num_scheduled_tokens],
    scheduler_output,
)
```

**步骤12: 生成投机token** (line 1385-1493)
```python
if self.speculative_config:
    if self.speculative_config.method == "ngram":
        spec_token_ids = self.generate_draft_token_ids(...)
    elif self.speculative_config.method == "medusa":
        spec_token_ids = self.drafter.propose(
            target_hidden_states=hidden_states, ...)
    elif self.speculative_config.use_eagle():
        spec_token_ids = self.drafter.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            ...)
```

**步骤13: 返回输出** (line 1499-1508)
```python
return ModelRunnerOutput(
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

### 6. Request (v1/request.py:19-194)

**职责**: V1请求对象，替代V0的Sequence/SequenceGroup

**关键属性**:
```python
self.request_id: str
self.prompt_token_ids: list[int]
self._output_token_ids: list[int]
self._all_token_ids: list[int]            # prompt + output
self.spec_token_ids: list[int]            # 投机token
self.num_computed_tokens: int             # 已计算token数
self.status: RequestStatus                # 状态

# 多模态
self.mm_positions: list[PlaceholderRange]
self.mm_inputs: list[MultiModalKwargs]
self.mm_hashes: list[str]

# 采样
self.sampling_params: SamplingParams
self.eos_token_id: Optional[int]

# 结构化输出
self.structured_output_request: Optional[StructuredOutputRequest]

# KV传输
self.kv_transfer_params: Optional[dict[str, Any]]
```

**关键属性**:
```python
@property
def num_tokens(self) -> int:
    """总token数: prompt + output"""
    return len(self._all_token_ids)

@property
def num_tokens_with_spec(self) -> int:
    """包括投机token的总数"""
    return len(self._all_token_ids) + len(self.spec_token_ids)
```

**与V0的Sequence/SequenceGroup对比**:
- V1单个`Request`类 vs V0的`Sequence` + `SequenceGroup`
- V1的`Request`直接支持投机token (`spec_token_ids`)
- V1的`Request`内置多模态支持
- V1的`Request`有更丰富的状态 (`RequestStatus`)

---

## 完整数据流

### 请求阶段

```
用户输入
    ↓
prompts: List[str] | List[Dict]
sampling_params: SamplingParams
    ↓
LLM.generate()
    ↓
LLM._validate_and_add_requests()
    ↓
    FOR EACH prompt:
        ↓
    LLMEngine.add_request(
        request_id: str,
        prompt: PromptType,
        params: SamplingParams
    )
        ↓
    Processor.preprocess()  # V1特有
        ↓
    EngineCoreRequest {
        request_id: str,
        prompt_token_ids: list[int],
        mm_inputs: Optional[list[MultiModalKwargs]],
        mm_hashes: Optional[list[str]],
        mm_placeholders: Optional[list[PlaceholderRange]],
        sampling_params: SamplingParams,
    }
        ↓
    engine_core.add_request(request)  # 异步添加
        ↓
    内部转换为 Request 对象
        ↓
    Scheduler.add_request(request)
        ↓
    添加到 waiting 队列
```

### 执行阶段 (循环)

```
WHILE has_unfinished_requests():
    ↓
LLMEngine.step()
    ↓
engine_core.get_output()  # 从异步EngineCore获取
    ↓
    ┌─────────────────────────────────────┐
    │ EngineCore内部执行 (异步/独立进程)   │
    └─────────────────────────────────────┘
    ↓
    ┌─────────────────────────────────────┐
    │ 阶段1: 调度 (Scheduler.schedule)     │
    └─────────────────────────────────────┘
    ↓
    FOR request IN running:
        计算 num_new_tokens = num_tokens_with_spec - num_computed_tokens

        IF request.has_encoder_inputs:
            调度编码器输入

        分配 KV cache:
            new_blocks = kv_cache_manager.allocate_slots(
                request, num_new_tokens, ...)

        IF new_blocks is None:
            抢占最低优先级请求
            preempt(lowest_priority_request)
        ELSE:
            scheduled_running_reqs.append(request)
            num_scheduled_tokens[req_id] = num_new_tokens
    ↓
    FOR request IN waiting:
        IF status == WAITING_FOR_REMOTE_KVS:
            检查KV传输是否完成
            IF not ready: CONTINUE

        IF status == WAITING_FOR_FSM:
            检查结构化输出FSM是否就绪
            IF not ready: CONTINUE

        IF num_computed_tokens == 0:
            # 前缀缓存
            computed_blocks, num_cached = \
                kv_cache_manager.get_computed_blocks(request)

            # KV传输 (P/D)
            IF connector:
                num_external, load_async = \
                    connector.get_num_new_matched_tokens(...)

        分配 KV cache:
            new_blocks = kv_cache_manager.allocate_slots(...)

        IF new_blocks:
            waiting.remove(request)
            running.append(request)
            request.status = RUNNING
            scheduled_new_reqs.append(request)
    ↓
    构建 SchedulerOutput {
        scheduled_new_reqs: List[NewRequestData],
        scheduled_cached_reqs: List[CachedRequestData],
        num_scheduled_tokens: Dict[str, int],
        total_num_scheduled_tokens: int,
        scheduled_spec_decode_tokens: Dict[str, List[int]],
        scheduled_encoder_inputs: Dict[str, List[int]],
        grammar_bitmask: Optional[np.ndarray],
    }
    ↓
    更新 num_computed_tokens:
        FOR req_id, num_tokens IN num_scheduled_tokens.items():
            requests[req_id].num_computed_tokens += num_tokens
    ↓
    ┌─────────────────────────────────────┐
    │ 阶段2: 模型执行                      │
    └─────────────────────────────────────┘
    Worker.execute_model(scheduler_output)
        ↓
    GPUModelRunner.execute_model(scheduler_output)
        ↓
        步骤1: _update_states(scheduler_output)
            更新 input_batch 和 requests cache
        ↓
        步骤2: _prepare_inputs(scheduler_output)
            ↓
            构建 input_ids, positions, slot_mapping
            构建 attn_metadata (per KV cache group)
            计算 logits_indices
            IF 投机解码:
                构建 spec_decode_metadata
        ↓
        步骤3: 多模态编码器 (如果适用)
            ↓
            _execute_mm_encoder(scheduler_output)
                ↓
                批处理 mm_inputs
                ↓
                model.get_multimodal_embeddings(**mm_inputs)
                ↓
                缓存编码器输出到 self.encoder_cache
            ↓
            _gather_mm_embeddings(scheduler_output)
                ↓
                从 encoder_cache 收集需要的embeddings
        ↓
        步骤4: 准备模型输入
            ↓
            IF multimodal_model:
                inputs_embeds = model.get_input_embeddings(
                    input_ids, mm_embeds)
                input_ids = None
            ELSE:
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None

            positions = self.positions[:num_input_tokens]
                OR
            positions = self.mrope_positions[:, :num_input_tokens]  # M-RoPE
        ↓
        步骤5: 模型forward
            ↓
            with set_forward_context(attn_metadata, ...):
                model_output = model(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                )
            ↓
            IF pipeline_parallel:
                send/recv intermediate_tensors
            ↓
            hidden_states: Tensor[num_tokens, hidden_size]
        ↓
        步骤6: 计算logits
            ↓
            sample_hidden_states = hidden_states[logits_indices]
            logits = model.compute_logits(sample_hidden_states, None)
            ↓
            logits: Tensor[num_sampled_positions, vocab_size]
        ↓
        步骤7: 结构化输出 (如果适用)
            ↓
            apply_grammar_bitmask(scheduler_output, logits)
                修改 logits[grammar_constrained_positions]
        ↓
        步骤8: 采样
            ↓
            sampling_metadata = input_batch.sampling_metadata
            ↓
            IF spec_decode_metadata is None:
                # 标准采样
                sampler_output = sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            ELSE:
                # 投机解码采样
                bonus_logits = logits[bonus_logits_indices]
                sampler_output = sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # 拒绝采样
                target_logits = logits[target_logits_indices]
                output_token_ids = rejection_sampler(
                    spec_decode_metadata,
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
                sampler_output.sampled_token_ids = output_token_ids
            ↓
            sampled_token_ids: List[List[int]]  # [num_reqs, num_tokens_per_req]
        ↓
        步骤9: 过滤部分prefill的采样
            ↓
            FOR req_id IN input_batch.req_ids:
                seq_len = (req_state.num_computed_tokens +
                           num_scheduled_tokens[req_id])
                IF seq_len < req_state.num_tokens:
                    # 部分prefill，忽略采样
                    valid_sampled_token_ids[i].clear()
        ↓
        步骤10: 计算prompt logprobs (如果需要)
            ↓
            prompt_logprobs_dict = _get_prompt_logprobs_dict(
                hidden_states, scheduler_output)
        ↓
        步骤11: 生成投机token (如果启用)
            ↓
            IF speculative_config:
                IF method == "ngram":
                    spec_token_ids = drafter.propose(token_ids_cpu)
                ELIF method == "medusa":
                    spec_token_ids = drafter.propose(
                        target_hidden_states=hidden_states)
                ELIF method == "eagle":
                    spec_token_ids = drafter.propose(
                        target_token_ids=target_token_ids,
                        target_positions=target_positions,
                        target_hidden_states=target_hidden_states,
                        next_token_ids=next_token_ids,
                    )
        ↓
        返回 ModelRunnerOutput {
            req_ids: List[str],
            req_id_to_index: Dict[str, int],
            sampled_token_ids: List[List[int]],
            spec_token_ids: Optional[List[List[int]]],
            logprobs: Optional[List[...]],
            prompt_logprobs_dict: Dict[str, LogprobsTensors],
            finished_sending: Optional[Set[str]],  # KV传输
            finished_recving: Optional[Set[str]],  # KV传输
        }
    ↓
    ┌─────────────────────────────────────┐
    │ 阶段3: 处理输出                      │
    └─────────────────────────────────────┘
    Scheduler.update_from_output(
        scheduler_output,
        model_runner_output,
    )
        ↓
        FOR request IN running:
            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]

            # 投机解码: 处理拒绝
            IF scheduled_spec_token_ids:
                num_rejected = (len(scheduled_spec_token_ids) + 1 -
                                len(generated_token_ids))
                request.num_computed_tokens -= num_rejected

            # 添加token
            FOR token_id IN generated_token_ids:
                request.append_output_token_ids(token_id)

                # 检查停止
                stopped = check_stop(request, max_model_len)
                IF stopped:
                    _free_request(request)
                    BREAK

            # 添加新的投机token
            IF spec_token_ids:
                request.spec_token_ids = spec_token_ids[req_index]

            # 创建输出
            outputs[client_index].append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=generated_token_ids,
                    finish_reason=request.get_finished_reason(),
                    new_logprobs=new_logprobs,
                ))
        ↓
        返回 Dict[int, EngineCoreOutputs] {
            client_index: EngineCoreOutputs {
                outputs: List[EngineCoreOutput],
                finished_requests: Optional[Set[str]],
                scheduler_stats: Optional[SchedulerStats],
            }
        }
    ↓
    (EngineCore内部执行结束)
    ↓
EngineCoreOutputs 返回给 engine_core
    ↓
LLMEngine.step() 从 engine_core.get_output() 获取
    ↓
output_processor.process_outputs(outputs)
    ↓
    FOR output IN outputs:
        IF output.finish_reason:
            finalize request

        detokenize(output.new_token_ids)
        ↓
        text: str
    ↓
    返回 RequestOutput {
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        outputs: List[CompletionOutput] {
            text: str,
            token_ids: List[int],
            cumulative_logprob: float,
            logprobs: Optional[...],
            finish_reason: Optional[str],
        },
        finished: bool,
    }
```

---

## 详细执行流程

### 1. 初始化流程

```python
# 用户创建LLM实例 (与V0类似)
llm = LLM(
    model="Qwen/Qwen2.5-7B",
    tensor_parallel_size=1,
    dtype="auto",
)

# V1内部执行:
# 1. 创建EngineArgs和VllmConfig (与V0相同)
# 2. 创建LLMEngine
engine = LLMEngine(
    vllm_config=vllm_config,
    executor_class=GPUExecutor,
    ...)

# 3. LLMEngine初始化
# 3.1 创建Processor (V1特有)
self.processor = Processor(
    model_config=model_config,
    lora_config=lora_config,
    tokenizer=tokenizer,
)

# 3.2 创建EngineCoreClient (V1核心创新)
self.engine_core = make_engine_core_client(
    vllm_config,
    executor_class,
    ...)
# EngineCoreClient可以是:
# - EngineCoreProc (多进程, 默认)
# - EngineCoreThreadedQueue (多线程)
# - EngineCore (同步, 测试用)

# 3.3 EngineCoreClient内部创建EngineCore
engine_core = EngineCore(vllm_config, executor_class, ...)
    ↓
    # 3.3.1 创建Executor和Worker
    executor = GPUExecutor(vllm_config)
    executor.initialize()
        ↓
        FOR each worker:
            worker.init_device()
            worker.load_model()
            worker.determine_available_memory()

    # 3.3.2 获取KV cache spec
    kv_cache_spec = executor.get_kv_cache_spec()
    # 返回: Dict[layer_name, KVCacheSpec]

    # 3.3.3 创建KVCacheConfig
    kv_cache_config = create_kv_cache_config(
        kv_cache_spec,
        num_gpu_blocks=available_memory // block_size,
    )

    # 3.3.4 初始化KV cache
    executor.initialize_from_config(kv_cache_config)

    # 3.3.5 创建Scheduler
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=...,
    )
    # Scheduler内部创建:
    # - KVCacheManager
    # - EncoderCacheManager
    # - KVConnector (如果启用)

    # 3.3.6 编译/warm up
    executor.compile_or_warm_up_model()
```

### 2. KV Cache初始化 (V1)

```python
# Worker.determine_available_memory()
# 与V0类似，执行profile run

# Worker.initialize_from_config(kv_cache_config)
model_runner.initialize_kv_cache(kv_cache_config)
    ↓
    # 1. Re-initialize input_batch (如果block_sizes不同)
    may_reinitialize_input_batch(kv_cache_config)

    # 2. 初始化attention backend
    initialize_attn_backend(kv_cache_config)
        ↓
        FOR kv_cache_group IN kv_cache_groups:
            attn_backend = get_attn_backend(
                head_size, dtype, block_size, use_mla, ...)
            attn_metadata_builder = attn_backend.get_builder_cls()(...)
            self.attn_backends.append(attn_backend)
            self.attn_metadata_builders.append(attn_metadata_builder)

    # 3. 初始化KV cache tensors
    kv_caches = initialize_kv_cache_tensors(kv_cache_config)
        ↓
        # 3.1 分配raw tensors
        FOR kv_cache_tensor IN kv_cache_tensors:
            tensor = torch.zeros(size, dtype=torch.int8, device="cuda")
            FOR layer_name IN shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        # 3.2 Reshape tensors
        FOR layer_name, kv_cache_group IN kv_cache_groups:
            raw_tensor = kv_cache_raw_tensors[layer_name]
            kv_cache_shape = attn_backend.get_kv_cache_shape(
                num_blocks, block_size, num_kv_heads, head_size)
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
            kv_caches[layer_name] = raw_tensor.view(dtype).view(
                kv_cache_shape).permute(*inv_order)

        # 3.3 Cross-layer KV sharing (如果启用)
        IF shared_kv_cache_layers:
            initialize_kv_cache_for_kv_sharing(...)

        # 3.4 Bind KV cache
        bind_kv_cache(kv_caches, static_forward_context, self.kv_caches)
```

### 3. 请求添加详细流程 (V1)

```python
# LLMEngine.add_request()

# 步骤1: 预处理输入
preprocessed = self.processor.preprocess(
    prompt="你好，请介绍一下vLLM",
    request_id="0",
    tokenizer=self.tokenizer,
    ...
)
# 返回: EngineCoreRequest

# 步骤2: 异步添加到EngineCore
self.engine_core.add_request(preprocessed)
    ↓
    # EngineCoreClient将请求发送到EngineCore进程/线程
    # EngineCore.add_request()
    ↓
    # 步骤2.1: 转换为Request对象
    request = Request.from_engine_core_request(engine_core_request)
    # Request {
    #     request_id: "0",
    #     prompt_token_ids: [1, 108, 100, 2500, ...],  # 长度: 15
    #     _output_token_ids: [],
    #     _all_token_ids: [1, 108, 100, 2500, ...],
    #     spec_token_ids: [],
    #     num_computed_tokens: 0,
    #     status: RequestStatus.WAITING,
    #     sampling_params: SamplingParams(...),
    # }

    # 步骤2.2: 添加到Scheduler
    scheduler.add_request(request)
    # 内部操作:
    # - 添加到 waiting 队列
    # - requests[request_id] = request
```

### 4. 调度详细流程 (V1)

见 [Scheduler.schedule()](#31-schedule-line-158-584) 章节

关键要点:
1. **无prefill/decode区分**: 统一Token级调度
2. **支持chunked prefill**: `num_new_tokens`可能小于prompt长度
3. **前缀缓存**: 自动查找已缓存的token
4. **KV传输**: P/D架构支持远程KV加载
5. **结构化输出**: FSM编译和约束
6. **投机解码**: 自动调度投机token

### 5. 模型执行详细流程 (V1)

见 [GPUModelRunner.execute_model()](#51-execute_model-line-1171-1508) 章节

关键创新:
1. **多KV cache组**: 支持不同attention层使用不同cache配置
2. **多模态编码器**: 分离的编码器执行和缓存
3. **M-RoPE**: 支持多维位置编码 (Qwen2-VL)
4. **投机解码集成**: Ngram/EAGLE/Medusa原生支持
5. **结构化输出**: Grammar bitmask约束
6. **CUDA Graph**: 支持full_cuda_graph模式 (FA3)

### 6. 输出处理详细流程 (V1)

```python
# Scheduler.update_from_output()

FOR request IN running:
    req_index = model_runner_output.req_id_to_index[req_id]
    generated_token_ids = sampled_token_ids[req_index]

    # 处理投机解码拒绝
    IF scheduled_spec_token_ids:
        num_rejected = len(scheduled_spec_token_ids) + 1 - len(generated_token_ids)
        request.num_computed_tokens -= num_rejected
        # Example:
        # scheduled_spec_token_ids = [A, B, C]  (3个投机token)
        # generated_token_ids = [X, A]  (只接受了A)
        # num_rejected = 3 + 1 - 2 = 2  (拒绝了B, C和bonus token)

    # 添加生成的token
    FOR token_id IN generated_token_ids:
        request.append_output_token_ids(token_id)
        # 更新 _output_token_ids 和 _all_token_ids

        # 检查停止条件
        stopped = check_stop(request, max_model_len)
        # 检查:
        # 1. EOS token
        # 2. max_tokens
        # 3. stop strings
        # 4. 长度限制

        IF stopped:
            kv_transfer_params = _free_request(request)
            # 释放: KV cache, 编码器cache
            # 添加到 finished_req_ids
            BREAK

    # 更新编码器cache (释放已使用的)
    FOR input_id IN cached_encoder_input_ids:
        mm_positions = request.mm_positions[input_id]
        IF mm_positions.offset + mm_positions.length <= request.num_computed_tokens:
            # 编码器输出已存入decoder KV cache
            encoder_cache_manager.free_encoder_input(request, input_id)

    # 提取logprobs
    IF request.sampling_params.logprobs:
        new_logprobs = logprobs.slice(req_index, req_index + 1)

    # 结构化输出: 更新FSM状态
    IF structured_output_manager.should_advance(request):
        request.structured_output_request.grammar.accept_tokens(
            req_id, generated_token_ids)

    # 添加新的投机token
    IF spec_token_ids:
        IF structured_output_manager.should_advance(request):
            # 验证投机token是否符合语法
            request.spec_token_ids = metadata.grammar.validate_tokens(
                spec_token_ids[req_index])
        ELSE:
            request.spec_token_ids = spec_token_ids[req_index]

    # 获取prompt logprobs
    prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)

    # 创建输出
    IF generated_token_ids OR kv_transfer_params:
        outputs[request.client_index].append(
            EngineCoreOutput(
                request_id=req_id,
                new_token_ids=generated_token_ids,
                finish_reason=request.get_finished_reason(),
                new_logprobs=new_logprobs,
                new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                stop_reason=request.stop_reason,
                events=request.take_events(),
                kv_transfer_params=kv_transfer_params,
                num_cached_tokens=request.num_cached_tokens,
            ))

    IF not stopped:
        new_running.append(request)

# 更新running队列
self.running = new_running

# 创建EngineCoreOutputs
engine_core_outputs = {
    client_index: EngineCoreOutputs(
        outputs=outputs_for_client,
        finished_requests=finished_set,
        scheduler_stats=stats,
    )
    FOR client_index, outputs_for_client IN outputs.items()
}

RETURN engine_core_outputs
```

---

## 关键数据类型

### 1. EngineCoreRequest
```python
@dataclass
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int]
    mm_inputs: Optional[list[MultiModalKwargs]]
    mm_hashes: Optional[list[str]]
    mm_placeholders: Optional[list[PlaceholderRange]]
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]
```

### 2. Request (v1/request.py)
```python
class Request:
    request_id: str
    client_index: int                          # 多客户端支持

    # Token数据
    prompt_token_ids: list[int]
    _output_token_ids: list[int]
    _all_token_ids: list[int]                  # prompt + output
    spec_token_ids: list[int]                  # 投机token
    num_computed_tokens: int                   # 已计算token数

    # 状态
    status: RequestStatus                      # WAITING, RUNNING, etc.
    events: list[EngineCoreEvent]              # 性能事件
    stop_reason: Union[int, str, None]

    # 采样
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    max_tokens: int

    # 多模态
    mm_positions: list[PlaceholderRange]
    mm_inputs: list[MultiModalKwargs]
    mm_hashes: list[str]
    has_encoder_inputs: bool

    # 结构化输出
    structured_output_request: Optional[StructuredOutputRequest]

    # KV传输
    kv_transfer_params: Optional[dict[str, Any]]

    # 前缀缓存
    num_cached_tokens: int                     # -1表示未缓存
    cache_salt: Optional[str]

    # 属性
    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)
```

### 3. RequestStatus
```python
class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()                      # 等待调度
    WAITING_FOR_FSM = enum.auto()              # 等待FSM编译 (结构化输出)
    WAITING_FOR_REMOTE_KVS = enum.auto()       # 等待KV传输
    RUNNING = enum.auto()                      # 运行中
    PREEMPTED = enum.auto()                    # 被抢占
    FINISHED_STOPPED = enum.auto()             # 正常结束 (EOS)
    FINISHED_LENGTH_CAPPED = enum.auto()       # 达到长度上限
    FINISHED_ABORTED = enum.auto()             # 被中止
    FINISHED_IGNORED = enum.auto()             # 被忽略
```

### 4. SchedulerOutput (v1/core/sched/output.py)
```python
@dataclass
class SchedulerOutput:
    # 调度的请求
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: list[CachedRequestData]

    # Token统计
    num_scheduled_tokens: dict[str, int]       # req_id -> num_tokens
    total_num_scheduled_tokens: int

    # 投机解码
    scheduled_spec_decode_tokens: dict[str, list[int]]

    # 编码器调度
    scheduled_encoder_inputs: dict[str, list[int]]

    # Cascade attention
    num_common_prefix_blocks: list[int]        # 每个KV cache组

    # 完成的请求
    finished_req_ids: set[str]
    free_encoder_input_ids: list[tuple[str, int]]

    # 结构化输出
    structured_output_request_ids: dict[str, int]
    grammar_bitmask: Optional[np.ndarray]

    # KV传输
    kv_connector_metadata: Optional[KVConnectorMetadata]
```

### 5. NewRequestData
```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    mm_inputs: list[MultiModalKwargs]
    mm_hashes: list[str]
    mm_positions: list[PlaceholderRange]
    sampling_params: SamplingParams
    block_ids: tuple[list[int], ...]           # 每个KV cache组
    num_computed_tokens: int
    lora_request: Optional[LoRARequest]
```

### 6. CachedRequestData
```python
@dataclass
class CachedRequestData:
    req_id: str
    resumed_from_preemption: bool              # 从抢占恢复
    new_token_ids: list[int]                   # 新生成的token
    new_block_ids: tuple[list[int], ...]       # 新分配的块
    num_computed_tokens: int
```

### 7. ModelRunnerOutput (v1/outputs.py)
```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]         # [num_reqs, num_tokens]
    spec_token_ids: Optional[list[list[int]]]  # 投机token
    logprobs: Optional[list[...]]
    prompt_logprobs_dict: dict[str, LogprobsTensors]

    # KV传输
    finished_sending: Optional[set[str]]       # 完成发送的请求
    finished_recving: Optional[set[str]]       # 完成接收的请求
```

### 8. EngineCoreOutput
```python
@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    finish_reason: Optional[FinishReason]
    new_logprobs: Optional[...]
    new_prompt_logprobs_tensors: Optional[LogprobsTensors]
    stop_reason: Union[int, str, None]
    events: Optional[list[EngineCoreEvent]]
    kv_transfer_params: Optional[dict[str, Any]]
    num_cached_tokens: int
```

### 9. EngineCoreOutputs
```python
@dataclass
class EngineCoreOutputs:
    outputs: list[EngineCoreOutput]
    finished_requests: Optional[set[str]]      # 完成的请求ID
    scheduler_stats: Optional[SchedulerStats]
    timestamp: float
```

### 10. KVCacheBlocks
```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[list[KVCacheBlock], ...]
    # blocks[i][j] = i-th KV cache组的第j个块

    def get_block_ids(self) -> tuple[list[int], ...]:
        return tuple([blk.block_id for blk in group]
                     for group in self.blocks)
```

### 11. KVCacheBlock
```python
@dataclass
class KVCacheBlock:
    block_id: int                              # 物理块ID
    block_hash: Optional[BlockHash]            # 前缀缓存哈希
    ref_cnt: int                               # 引用计数
```

### 12. AttentionMetadata (per backend)
```python
# FlashAttention backend示例
@dataclass
class FlashAttentionMetadata:
    query_start_loc: torch.Tensor              # [num_reqs + 1]
    seq_lens: torch.Tensor                     # [num_reqs]
    max_query_len: int
    max_kv_len: int
    slot_mapping: torch.Tensor                 # [num_tokens]
    block_table: torch.Tensor                  # [num_reqs, max_blocks]

    # Cascade attention
    common_prefix_len: int
```

### 13. SpecDecodeMetadata
```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor              # 投机token IDs
    num_draft_tokens: list[int]                # 每个请求的投机token数
    cu_num_draft_tokens: torch.Tensor          # 累积和
    target_logits_indices: torch.Tensor        # 目标logits位置
    bonus_logits_indices: torch.Tensor         # bonus logits位置
    logits_indices: torch.Tensor               # 所有logits位置
```

---

## V0与V1对比

| 特性 | V0架构 | V1架构 |
|------|--------|--------|
| **执行模式** | 同步 | 异步/多进程 |
| **调度粒度** | Prefill/Decode分阶段 | 统一Token级调度 |
| **请求抽象** | Sequence + SequenceGroup | Request |
| **KV Cache管理** | BlockSpaceManager | KVCacheManager + Coordinator |
| **前缀缓存** | Block级哈希 | 增强的Block哈希 + LRU |
| **投机解码** | 外部集成 | 内置支持 (Ngram/EAGLE/Medusa) |
| **多模态** | 基础支持 | EncoderCacheManager专门优化 |
| **结构化输出** | 有限支持 | StructuredOutputManager + FSM |
| **KV传输** | 不支持 | 原生支持 (P/D分离) |
| **Chunked Prefill** | 基础支持 | 无缝集成 |
| **调度器状态** | WAITING, RUNNING, SWAPPED | WAITING, WAITING_FOR_FSM, WAITING_FOR_REMOTE_KVS, RUNNING, PREEMPTED, FINISHED_* |
| **输出类型** | SamplerOutput | ModelRunnerOutput |
| **编码器缓存** | 临时存储 | EncoderCacheManager管理 |
| **Pipeline并行** | 支持 | 支持 + broadcast优化 |
| **CUDA Graph** | Piecewise | Piecewise + Full (FA3) |

### 架构对比图

```
V0:
LLM → LLMEngine → Scheduler → Executor → Worker → ModelRunner → Model
      ↓
    (同步调用链)

V1:
LLM → LLMEngine → EngineCoreClient → EngineCore
      ↓                                   ↓
    (异步)                        Scheduler + KVCacheManager
                                        ↓
                                    Executor → Worker → GPUModelRunner
                                                            ↓
                                                      Model + Sampler + Drafter
```

### 数据结构对比

| V0 | V1 |
|----|---|
| Sequence | Request |
| SequenceGroup | (合并到Request) |
| SequenceData | Request的内部属性 |
| BlockSpaceManager | KVCacheManager |
| SchedulerOutputs | SchedulerOutput |
| SamplerOutput | ModelRunnerOutput |
| RequestOutput | RequestOutput (类似) |

### 性能优化对比

| 优化 | V0 | V1 |
|------|----|----|
| 前缀缓存 | ✓ | ✓✓ (增强) |
| Chunked prefill | ✓ | ✓✓ (无缝) |
| 投机解码 | 外部 | ✓✓ (内置) |
| Cascade attention | ✓ | ✓ |
| CUDA Graph | Piecewise | Piecewise + Full |
| 多模态优化 | 基础 | ✓✓ (专门优化) |
| KV传输 | ✗ | ✓✓ |
| 结构化输出 | 有限 | ✓✓ |
| Data parallelism | 基础 | ✓✓ (padding优化) |

---

## 总结

### V1架构特点

1. **统一调度**: Token级调度，无prefill/decode区分
2. **异步执行**: EngineCoreClient抽象，支持多进程/多线程
3. **增强缓存**: KVCacheManager + Coordinator，更智能的前缀缓存
4. **内置投机**: Ngram/EAGLE/Medusa原生支持
5. **多模态优化**: EncoderCacheManager专门处理视觉编码器
6. **KV传输**: 支持P/D分离和卸载
7. **结构化输出**: FSM编译和Grammar约束
8. **灵活扩展**: 支持多客户端、多KV cache组

### 性能关键点

1. **调度效率**: 统一Token级调度，无阶段切换开销
2. **缓存命中率**: 增强的前缀缓存算法
3. **投机解码加速**: 内置drafter和rejection sampler
4. **编码器复用**: EncoderCacheManager避免重复计算
5. **CUDA Graph**: 支持full_cuda_graph (FA3)
6. **并行优化**: Data/Tensor/Pipeline并行 + 序列并行

### 适用场景

- **高吞吐离线推理**: 最大化GPU利用率
- **多模态应用**: 视觉语言模型 (VLM)
- **投机解码**: 加速长文本生成
- **P/D分离**: Prefill和Decode分离部署
- **结构化输出**: JSON/Grammar约束生成
- **前缀缓存**: 大量共享前缀的场景

---

## 附录: 关键代码位置

| 组件 | 文件路径 | 行号 |
|------|---------|------|
| LLMEngine.add_request | v1/engine/llm_engine.py | 116-165 |
| LLMEngine.step | v1/engine/llm_engine.py | 224-250 |
| Scheduler.schedule | v1/core/sched/scheduler.py | 158-584 |
| Scheduler.update_from_output | v1/core/sched/scheduler.py | 700-865 |
| KVCacheManager.get_computed_blocks | v1/core/kv_cache_manager.py | 133-180 |
| KVCacheManager.allocate_slots | v1/core/kv_cache_manager.py | 182-291 |
| GPUModelRunner.execute_model | v1/worker/gpu_model_runner.py | 1171-1508 |
| GPUModelRunner._prepare_inputs | v1/worker/gpu_model_runner.py | 552-726 |
| GPUModelRunner._execute_mm_encoder | v1/worker/gpu_model_runner.py | 938-1001 |
| Worker.execute_model | v1/worker/gpu_worker.py | 283-303 |
| Request | v1/request.py | 19-194 |
| SchedulerOutput | v1/core/sched/output.py | 110-155 |
