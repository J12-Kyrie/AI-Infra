# vLLM V0 架构详解 - Qwen3 离线推理流程

## 目录
1. [架构概览](#架构概览)
2. [核心组件](#核心组件)
3. [完整数据流](#完整数据流)
4. [详细执行流程](#详细执行流程)
5. [关键数据类型](#关键数据类型)

---

## 架构概览

vLLM V0 架构采用**同步执行**模式，主要用于离线批处理推理。核心设计理念是通过高效的调度和内存管理（PagedAttention）最大化GPU利用率。

### 架构层次
```
用户层 (LLM)
    ↓
引擎层 (LLMEngine)
    ↓
调度层 (Scheduler)
    ↓
执行层 (Executor → Worker → ModelRunner)
    ↓
模型层 (Model + KV Cache)
```

---

## 核心组件

### 1. LLM (entrypoints/llm.py:59-1545)

**职责**: 用户API接口，封装LLMEngine提供简洁的推理接口

**关键方法**:
```python
def __init__(
    model: str,
    tokenizer: Optional[str] = None,
    tensor_parallel_size: int = 1,
    dtype: ModelDType = "auto",
    gpu_memory_utilization: float = 0.9,
    ...
) -> None
```

```python
def generate(
    prompts: Union[PromptType, Sequence[PromptType]],
    sampling_params: Optional[Union[SamplingParams, Sequence[SamplingParams]]] = None,
    use_tqdm: bool = True,
    lora_request: Optional[LoRARequest] = None,
) -> list[RequestOutput]
```

**核心流程**:
1. 初始化 LLMEngine (line 243-244)
2. 验证并添加请求 (line 464-472)
3. 循环执行 engine.step() (line 474)
4. 收集并返回结果 (line 475)

---

### 2. LLMEngine (engine/llm_engine.py:123-2093)

**职责**: 核心调度引擎，协调Scheduler、Executor、Tokenizer

**关键属性**:
```python
self.model_config: ModelConfig       # 模型配置
self.scheduler: List[Scheduler]      # 调度器（支持pipeline并行）
self.model_executor: ExecutorBase    # 模型执行器
self.tokenizer: TokenizerGroup       # 分词器
self.detokenizer: Detokenizer        # 反分词器
self.output_processor: SequenceGroupOutputProcessor  # 输出处理器
```

**核心方法**:

#### 2.1 add_request (line 631-730)
```python
def add_request(
    request_id: str,
    prompt: PromptType,
    params: Union[SamplingParams, PoolingParams],
    arrival_time: Optional[float] = None,
    lora_request: Optional[LoRARequest] = None,
    ...
) -> None
```

**执行流程**:
1. **预处理输入** (line 714-719):
   ```python
   processed_inputs = self.input_preprocessor.preprocess(
       prompt,
       tokenization_kwargs=tokenization_kwargs,
       lora_request=lora_request,
   )
   # 返回: ProcessorInputs (包含 prompt_token_ids, multi_modal_data 等)
   ```

2. **创建序列** (line 577-590):
   ```python
   # 创建 Sequence 对象
   seq = Sequence(
       seq_id=next(self.seq_counter),
       inputs=decoder_inputs,
       block_size=self.cache_config.block_size,
       eos_token_id=eos_token_id,
   )
   ```

3. **创建 SequenceGroup** (line 593-616):
   ```python
   if isinstance(params, SamplingParams):
       seq_group = self._create_sequence_group_with_sampling(...)
   # 返回: SequenceGroup (包含 seqs, sampling_params, metrics 等)
   ```

4. **添加到调度器** (line 618-624):
   ```python
   min_cost_scheduler.add_seq_group(seq_group)
   ```

#### 2.2 step (line 1212-1440)

**职责**: 执行一次迭代，完成调度→执行→输出处理

**执行流程**:

**阶段1: 调度** (line 1293-1296)
```python
(seq_group_metadata_list, scheduler_outputs, allow_async_output_proc
) = self.scheduler[virtual_engine].schedule()
```

**返回类型**: `SchedulerOutputs`
```python
@dataclass
class SchedulerOutputs:
    scheduled_seq_groups: List[ScheduledSequenceGroup]  # 要执行的序列组
    num_batched_tokens: int                              # 批次token数
    blocks_to_swap_in: List[Tuple[int, int]]           # CPU→GPU交换
    blocks_to_swap_out: List[Tuple[int, int]]          # GPU→CPU交换
    blocks_to_copy: List[Tuple[int, int]]              # GPU内复制
    ignored_seq_groups: List[SequenceGroup]            # 被忽略的序列组
    num_lookahead_slots: int                           # 预留槽位数
```

**阶段2: 构建执行请求** (line 1335-1345)
```python
execute_model_req = ExecuteModelRequest(
    seq_group_metadata_list=seq_group_metadata_list,
    blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
    blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
    blocks_to_copy=scheduler_outputs.blocks_to_copy,
    num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
)
```

**阶段3: 执行模型** (line 1352-1353)
```python
outputs = self.model_executor.execute_model(
    execute_model_req=execute_model_req
)
# 返回: List[SamplerOutput]
```

**阶段4: 处理输出** (line 1397-1421)
```python
# 添加到输出队列
ctx.append_output(
    outputs=outputs,
    seq_group_metadata_list=seq_group_metadata_list,
    scheduler_outputs=scheduler_outputs,
    is_async=allow_async_output_proc,
    is_last_step=True,
)

# 处理模型输出
self._process_model_outputs(ctx=ctx)
```

**阶段5: 返回结果** (line 1440)
```python
return ctx.request_outputs  # List[RequestOutput]
```

---

### 3. Scheduler (core/scheduler.py)

**职责**: 请求调度和KV cache块管理

**关键数据结构**:
```python
self.waiting: Deque[SequenceGroup]    # 等待队列
self.running: List[SequenceGroup]     # 运行队列
self.swapped: List[SequenceGroup]     # 被换出的队列
self.block_manager: BlockSpaceManager # KV cache块管理器
```

**调度策略**:
1. **Prefill优先**: 优先调度新请求的prefill阶段
2. **内存感知**: 根据KV cache可用块数调度
3. **抢占机制**: 内存不足时可抢占低优先级请求

**schedule() 输出**:
```python
SchedulerOutputs(
    scheduled_seq_groups=[...],    # 本次执行的序列组
    num_batched_tokens=total_tokens,
    blocks_to_swap_in=[...],       # 内存管理操作
    blocks_to_swap_out=[...],
    blocks_to_copy=[...],
)
```

---

### 4. Worker (worker/worker.py:39-578)

**职责**: GPU上的实际执行单元，管理模型和KV cache

**关键属性**:
```python
self.model_runner: GPUModelRunnerBase    # 模型运行器
self.cache_engine: List[CacheEngine]     # KV cache引擎
self.gpu_cache: List[List[torch.Tensor]] # GPU KV cache
```

**核心方法**:

#### 4.1 execute_model (line 387-449)
```python
def execute_model(
    execute_model_req: Optional[ExecuteModelRequest] = None,
) -> Optional[List[SamplerOutput]]
```

**执行流程**:

**步骤1: 准备输入** (line 395-399)
```python
inputs = self.prepare_input(execute_model_req)
# 返回: Tuple[BroadcastableModelInput, WorkerInput, Dict]
model_input, worker_input, kwargs = inputs
```

**步骤2: 执行Worker操作** (line 404)
```python
self.execute_worker(worker_input)
# 执行: KV cache swap_in, swap_out, copy操作
```

**步骤3: 接收中间张量** (line 412-415, Pipeline并行)
```python
if not get_pp_group().is_first_rank:
    intermediate_tensors = IntermediateTensors(
        get_pp_group().recv_tensor_dict()
    )
```

**步骤4: 执行模型** (line 421-428)
```python
output = self.model_runner.execute_model(
    model_input=model_input,
    kv_caches=self.kv_cache[worker_input.virtual_engine],
    intermediate_tensors=intermediate_tensors,
    num_steps=num_steps,
)
# 返回: List[SamplerOutput] 或 IntermediateTensors
```

**步骤5: 发送中间张量** (line 431-440, Pipeline并行)
```python
if not get_pp_group().is_last_rank:
    get_pp_group().send_tensor_dict(output.tensors)
    return [None]
```

---

### 5. ModelRunner (worker/model_runner.py)

**职责**: 准备模型输入、执行forward、采样

**关键方法**:

#### 5.1 prepare_model_input
```python
def prepare_model_input(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    virtual_engine: int = 0,
    finished_requests_ids: Optional[List[str]] = None,
) -> ModelRunnerInputBase
```

**构建**:
- `input_ids`: Token IDs
- `positions`: 位置编码
- `attn_metadata`: 注意力元数据
- `slot_mapping`: KV cache槽位映射

#### 5.2 execute_model
```python
def execute_model(
    model_input: ModelRunnerInputBase,
    kv_caches: List[torch.Tensor],
    intermediate_tensors: Optional[IntermediateTensors] = None,
    num_steps: int = 1,
) -> List[SamplerOutput]
```

**执行流程**:
1. 执行模型forward
2. 计算logits
3. 采样生成token
4. 返回SamplerOutput

---

### 6. CacheEngine (worker/cache_engine.py)

**职责**: 管理GPU/CPU上的KV cache块

**关键操作**:
```python
def swap_in(self, src_to_dst: torch.Tensor) -> None
    # CPU → GPU

def swap_out(self, src_to_dst: torch.Tensor) -> None
    # GPU → CPU

def copy(self, src_to_dsts: torch.Tensor) -> None
    # GPU内复制（用于beam search等）
```

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
    InputPreprocessor.preprocess()
        ↓
    ProcessorInputs {
        prompt_token_ids: List[int],
        multi_modal_data: Optional[MultiModalDataDict],
    }
        ↓
    创建 Sequence(
        seq_id: int,
        prompt_token_ids: List[int],
        block_size: int,
        eos_token_id: int
    )
        ↓
    创建 SequenceGroup(
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float
    )
        ↓
    Scheduler.add_seq_group(seq_group)
```

### 执行阶段 (循环)

```
WHILE has_unfinished_requests():
    ↓
LLMEngine.step()
    ↓
    ┌─────────────────────────────────────┐
    │ 阶段1: 调度                         │
    └─────────────────────────────────────┘
    Scheduler.schedule()
        ↓
        分析waiting/running/swapped队列
        ↓
        决定本次执行的序列组
        ↓
    SchedulerOutputs {
        scheduled_seq_groups: List[ScheduledSequenceGroup],
        num_batched_tokens: int,
        blocks_to_swap_in: List[Tuple[int, int]],
        blocks_to_swap_out: List[Tuple[int, int]],
        blocks_to_copy: List[Tuple[int, int]],
    }
    ↓
    ┌─────────────────────────────────────┐
    │ 阶段2: 构建执行请求                 │
    └─────────────────────────────────────┘
    创建 ExecuteModelRequest {
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: List[Tuple[int, int]],
        blocks_to_swap_out: List[Tuple[int, int]],
        blocks_to_copy: List[Tuple[int, int]],
    }
        ↓
        其中 SequenceGroupMetadata {
            request_id: str,
            is_prompt: bool,
            seq_data: Dict[int, SequenceData],
            sampling_params: SamplingParams,
            block_tables: Dict[int, List[int]],
            token_chunk_size: int,
        }
    ↓
    ┌─────────────────────────────────────┐
    │ 阶段3: 执行模型                     │
    └─────────────────────────────────────┘
    ModelExecutor.execute_model(execute_model_req)
        ↓
    Worker.execute_model()
        ↓
        步骤1: prepare_input()
            ↓
        WorkerInput {
            num_seq_groups: int,
            blocks_to_swap_in: torch.Tensor,
            blocks_to_swap_out: torch.Tensor,
            blocks_to_copy: torch.Tensor,
        }
        BroadcastableModelInput {
            input_tokens: torch.Tensor,      # [num_tokens]
            positions: torch.Tensor,          # [num_tokens]
            attn_metadata: AttentionMetadata,
            ...
        }
            ↓
        步骤2: execute_worker()
            ↓
            CacheEngine.swap_in()   # CPU→GPU
            CacheEngine.swap_out()  # GPU→CPU
            CacheEngine.copy()      # GPU内复制
            ↓
        步骤3: ModelRunner.execute_model()
            ↓
            准备输入张量:
            - input_ids: Tensor[num_tokens]
            - positions: Tensor[num_tokens]
            - kv_caches: List[Tensor]
            - attn_metadata: AttentionMetadata {
                seq_lens: Tensor[num_seqs],
                slot_mapping: Tensor[num_tokens],
                block_tables: Tensor[num_seqs, max_blocks],
              }
            ↓
            Model.forward(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
            )
                ↓
            hidden_states: Tensor[num_tokens, hidden_size]
                ↓
            logits = lm_head(hidden_states)
                ↓
            logits: Tensor[num_tokens, vocab_size]
                ↓
            Sampler.forward(
                logits=logits,
                sampling_metadata=SamplingMetadata
            )
                ↓
    List[SamplerOutput] {
        outputs: List[CompletionSequenceGroupOutput] [
            CompletionSequenceGroupOutput {
                samples: List[SequenceOutput] [
                    SequenceOutput {
                        output_token: int,
                        logprobs: Dict[int, Logprob],
                    }
                ]
            }
        ]
    }
    ↓
    ┌─────────────────────────────────────┐
    │ 阶段4: 处理输出                     │
    └─────────────────────────────────────┘
    LLMEngine._process_model_outputs(outputs)
        ↓
        FOR EACH seq_group:
            ↓
            OutputProcessor.process_outputs(
                seq_group,
                outputs
            )
                ↓
                更新 Sequence.output_token_ids
                更新 Sequence.cumulative_logprob
                检查停止条件
                    ↓
            IF seq_group.is_finished():
                ↓
                创建 RequestOutput {
                    request_id: str,
                    prompt: str,
                    prompt_token_ids: List[int],
                    outputs: List[CompletionOutput] [
                        CompletionOutput {
                            index: int,
                            text: str,
                            token_ids: List[int],
                            cumulative_logprob: float,
                            logprobs: Optional[List[Dict[int, Logprob]]],
                            finish_reason: Optional[str],
                        }
                    ],
                    finished: bool,
                }
                    ↓
                添加到 ctx.request_outputs
                    ↓
                Scheduler.free_finished_seq_groups()
    ↓
返回 List[RequestOutput]
```

---

## 详细执行流程

### 1. 初始化流程

```python
# 1. 用户创建LLM实例
llm = LLM(
    model="Qwen/Qwen2.5-7B",
    tensor_parallel_size=1,
    dtype="auto",
    gpu_memory_utilization=0.9,
)

# 内部执行:
# 1.1 创建 EngineArgs
engine_args = EngineArgs(
    model="Qwen/Qwen2.5-7B",
    ...
)

# 1.2 创建 VllmConfig
vllm_config = engine_args.create_engine_config()
# VllmConfig包含:
# - model_config: ModelConfig
# - cache_config: CacheConfig
# - parallel_config: ParallelConfig
# - scheduler_config: SchedulerConfig
# - device_config: DeviceConfig

# 1.3 创建 LLMEngine
self.llm_engine = LLMEngine.from_engine_args(engine_args)

# 1.4 LLMEngine初始化
# - 初始化tokenizer
# - 创建ModelExecutor
# - 初始化KV cache (determine_num_available_blocks)
# - 创建Scheduler
# - 初始化OutputProcessor
```

### 2. KV Cache 初始化

```python
# Worker.determine_num_available_blocks()
# 目标: 确定可分配的KV cache块数量

# 步骤1: 执行profile run
self.model_runner.profile_run()
    ↓
    执行一次完整的forward pass
    记录峰值内存使用

# 步骤2: 计算可用内存
total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
memory_for_current_instance = (
    total_gpu_memory * gpu_memory_utilization
)
available_kv_cache_memory = (
    memory_for_current_instance -
    (model_weight_memory + activation_memory)
)

# 步骤3: 计算块数量
cache_block_size = (
    block_size *              # e.g., 16
    num_kv_heads *           # e.g., 32
    head_size *              # e.g., 128
    num_layers *             # e.g., 32
    2 *                      # K和V
    dtype_size               # e.g., 2 bytes for fp16
)
num_gpu_blocks = available_kv_cache_memory // cache_block_size

# 步骤4: 分配KV cache张量
self.gpu_cache = [
    [
        torch.empty(
            (num_gpu_blocks, block_size, num_kv_heads, head_size),
            dtype=kv_cache_dtype,
            device="cuda"
        )
        for _ in range(2)  # K和V
    ]
    for _ in range(num_layers)
]
```

### 3. 请求添加详细流程

```python
# LLMEngine.add_request() 详细步骤

# 步骤1: 预处理输入
processed_inputs = self.input_preprocessor.preprocess(
    prompt="你好，请介绍一下vLLM",
    tokenization_kwargs={},
)
# 返回:
ProcessorInputs {
    "prompt_token_ids": [1, 108, 100, 2500, 3221, ...],  # tokenized
    "prompt": "你好，请介绍一下vLLM",
    "multi_modal_data": None,  # Qwen3是纯文本模型
}

# 步骤2: 创建Sequence
seq_id = next(self.seq_counter)  # e.g., 0
eos_token_id = self.tokenizer.eos_token_id  # e.g., 151643

seq = Sequence(
    seq_id=0,
    inputs=processed_inputs,
    block_size=16,
    eos_token_id=151643,
)
# Sequence内部状态:
seq.data = SequenceData(
    prompt_token_ids=[1, 108, 100, 2500, ...],  # 长度: 15
    output_token_ids=[],                         # 初始为空
    cumulative_logprob=0.0,
)
seq.logical_token_blocks = []  # 初始为空，调度时分配

# 步骤3: 创建SequenceGroup
seq_group = SequenceGroup(
    request_id="0",
    seqs=[seq],
    sampling_params=SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=100,
    ),
    arrival_time=time.time(),
)

# 步骤4: 添加到Scheduler
self.scheduler[0].add_seq_group(seq_group)
# 内部操作:
# - 添加到 waiting 队列
# - 等待下次schedule()调度
```

### 4. 调度详细流程

```python
# Scheduler.schedule() 详细步骤

# 当前状态:
# waiting: [seq_group_0]
# running: []
# swapped: []

# 步骤1: 处理waiting队列 (_schedule_prefills)
for seq_group in self.waiting:
    # 计算所需的块数
    num_prompt_tokens = seq_group.get_seqs()[0].get_len()  # 15
    num_blocks_needed = cdiv(num_prompt_tokens, block_size)  # ceil(15/16) = 1

    # 检查是否有足够的块
    num_free_blocks = self.block_manager.get_num_free_gpu_blocks()  # e.g., 1000

    if num_free_blocks >= num_blocks_needed:
        # 分配块
        block_table = self.block_manager.allocate(seq_group)
        # block_table: [0]  (分配了物理块0)

        # 移动到running队列
        self.waiting.remove(seq_group)
        self.running.append(seq_group)

        # 添加到scheduled列表
        scheduled_seq_groups.append(
            ScheduledSequenceGroup(
                seq_group=seq_group,
                token_chunk_size=15,  # prefill全部15个token
            )
        )
        num_batched_tokens += 15

# 步骤2: 处理running队列 (_schedule_running)
# (第一次step时running为空，跳过)

# 步骤3: 返回调度结果
return SchedulerOutputs(
    scheduled_seq_groups=[
        ScheduledSequenceGroup(
            seq_group=seq_group_0,
            token_chunk_size=15,
        )
    ],
    num_batched_tokens=15,
    num_prefill_groups=1,
    blocks_to_swap_in=[],
    blocks_to_swap_out=[],
    blocks_to_copy=[],
)
```

### 5. 模型执行详细流程

```python
# ModelRunner.execute_model() 详细步骤

# 输入: model_input, kv_caches

# 步骤1: 准备输入张量
input_ids = torch.tensor([1, 108, 100, 2500, ...], device="cuda")  # [15]
positions = torch.arange(0, 15, device="cuda")                      # [0,1,...,14]

# 步骤2: 准备注意力元数据
attn_metadata = AttentionMetadata(
    seq_lens=torch.tensor([15], device="cuda"),        # 序列长度
    max_seq_len=15,
    num_prefill_tokens=15,
    num_decode_tokens=0,
    slot_mapping=torch.tensor([0,1,2,...,14], device="cuda"),  # KV cache槽位
    block_tables=torch.tensor([[0]], device="cuda"),   # 物理块表
)

# 步骤3: 模型forward
# Qwen2ForCausalLM.forward()
hidden_states = self.model.forward(
    input_ids=input_ids,         # [15]
    positions=positions,          # [15]
    kv_caches=kv_caches,         # List[Tuple[Tensor, Tensor]] * num_layers
    attn_metadata=attn_metadata,
)
# 执行流程:
# 3.1 Embedding
embeds = self.embed_tokens(input_ids)  # [15, 4096]

# 3.2 Transformer Layers (循环32层)
for layer in self.layers:
    # Self-Attention
    hidden_states = layer.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        kv_cache=kv_caches[layer_idx],  # (K cache, V cache)
        attn_metadata=attn_metadata,
    )
    # 内部实现:
    # - 计算 Q, K, V
    # - 将 K, V 写入 kv_cache
    # - 执行 PagedAttention
    # - 返回 attention output

    # FFN
    hidden_states = layer.mlp(hidden_states)

# 3.3 Final LayerNorm
hidden_states = self.norm(hidden_states)  # [15, 4096]

# 步骤4: 计算logits
logits = self.lm_head(hidden_states)  # [15, vocab_size=151936]

# 步骤5: 采样
# 只对最后一个token采样 (因为是prefill)
last_token_logits = logits[-1:]  # [1, 151936]

sampler_output = self.sampler(
    logits=last_token_logits,
    sampling_metadata=SamplingMetadata(
        seq_groups=[seq_group],
        selected_token_indices=[0],  # 采样第0个位置
        temperature=torch.tensor([0.0]),
        ...
    )
)

# 返回:
return [
    SamplerOutput(
        outputs=[
            CompletionSequenceGroupOutput(
                samples=[
                    SequenceOutput(
                        output_token=12345,  # 采样的token ID
                        logprobs={
                            12345: Logprob(logprob=-0.001, rank=1),
                            23456: Logprob(logprob=-3.2, rank=2),
                            ...
                        }
                    )
                ]
            )
        ]
    )
]
```

### 6. 输出处理详细流程

```python
# LLMEngine._process_model_outputs() 详细步骤

# 输入: sampler_outputs (List[SamplerOutput])

# 步骤1: 提取采样的token
for i, seq_group_meta in enumerate(seq_group_metadata_list):
    seq_group = scheduler_outputs.scheduled_seq_groups[i].seq_group
    sampler_output = outputs[0]  # 第一个(也是唯一的)输出

    # 获取采样结果
    seq_output = sampler_output.outputs[i].samples[0]
    output_token = seq_output.output_token  # e.g., 12345
    logprobs = seq_output.logprobs

    # 步骤2: 更新Sequence状态
    seq = seq_group.get_seqs()[0]
    seq.append_token_id(
        token_id=12345,
        logprobs={12345: Logprob(-0.001, 1), ...}
    )
    # 现在 seq.data.output_token_ids = [12345]

    # 步骤3: 更新num_computed_tokens
    seq_group.update_num_computed_tokens(15)  # prefill了15个token

    # 步骤4: 检查停止条件
    if not seq_group.is_finished():
        # 未完成，继续下次迭代
        continue
    else:
        # 步骤5: 创建最终输出
        output_text = self.detokenizer.decode(
            seq.get_output_token_ids()  # [12345, ...]
        )

        request_output = RequestOutput(
            request_id="0",
            prompt="你好，请介绍一下vLLM",
            prompt_token_ids=[1, 108, 100, 2500, ...],
            outputs=[
                CompletionOutput(
                    index=0,
                    text=output_text,
                    token_ids=[12345, 23456, ...],
                    cumulative_logprob=-15.6,
                    logprobs=[...],
                    finish_reason="stop",  # 或 "length"
                )
            ],
            finished=True,
        )

        # 步骤6: 释放资源
        self.scheduler.free_seq(seq)
        # 释放分配的KV cache块

# 返回
return [request_output]
```

### 7. 后续迭代 (Decode阶段)

第一次step完成prefill后，后续每次step只处理1个token (decode):

```python
# 第2次 step():

# Scheduler.schedule()
# - running队列: [seq_group_0]
# - seq_group_0的num_computed_tokens=15, output_token_ids=[12345]

scheduled_seq_groups = [
    ScheduledSequenceGroup(
        seq_group=seq_group_0,
        token_chunk_size=1,  # decode只处理1个token
    )
]

# ModelRunner.execute_model()
input_ids = torch.tensor([12345], device="cuda")  # 上次生成的token
positions = torch.tensor([15], device="cuda")     # position=15

attn_metadata = AttentionMetadata(
    seq_lens=torch.tensor([16], device="cuda"),   # 现在序列长度=16
    num_prefill_tokens=0,
    num_decode_tokens=1,
    slot_mapping=torch.tensor([15], device="cuda"),  # 写入第15个槽位
    block_tables=torch.tensor([[0]], device="cuda"),
)

# forward → 采样 → 生成新token (e.g., 23456)
# 添加到 output_token_ids: [12345, 23456]

# 重复直到:
# - 生成 EOS token
# - 达到 max_tokens
# - 其他停止条件
```

---

## 关键数据类型

### 1. PromptType
```python
# 用户输入的prompt类型
PromptType = Union[
    str,                        # 文本: "你好"
    List[int],                  # Token IDs: [1, 108, 100]
    TextPrompt,                 # {"prompt": "你好"}
    TokensPrompt,               # {"prompt_token_ids": [1, 108, 100]}
]
```

### 2. SamplingParams
```python
@dataclass
class SamplingParams:
    n: int = 1                          # 生成数量
    temperature: float = 1.0            # 温度
    top_p: float = 1.0                  # nucleus采样
    top_k: int = -1                     # top-k采样
    max_tokens: Optional[int] = 16      # 最大生成token数
    stop: Optional[Union[str, List[str]]] = None  # 停止字符串
    logprobs: Optional[int] = None      # 返回logprobs数量
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
```

### 3. Sequence
```python
class Sequence:
    seq_id: int                         # 序列ID
    inputs: ProcessorInputs             # 输入数据
    block_size: int                     # KV cache块大小
    eos_token_id: Optional[int]         # EOS token ID

    data: SequenceData                  # 序列数据
    logical_token_blocks: List[LogicalTokenBlock]  # 逻辑块
```

### 4. SequenceData
```python
@dataclass
class SequenceData:
    prompt_token_ids: List[int]         # prompt的token IDs
    output_token_ids: List[int]         # 生成的token IDs
    cumulative_logprob: float           # 累积对数概率

    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_num_uncomputed_tokens(self) -> int:
        return len(self.get_token_ids()) - self.num_computed_tokens
```

### 5. SequenceGroup
```python
class SequenceGroup:
    request_id: str                     # 请求ID
    seqs: List[Sequence]                # 序列列表(用于beam search等)
    sampling_params: SamplingParams     # 采样参数
    arrival_time: float                 # 到达时间
    metrics: RequestMetrics             # 性能指标

    prompt_token_ids: List[int]         # prompt tokens (共享)

    def is_prefill(self) -> bool:
        """是否处于prefill阶段"""
        return self.seqs[0].get_num_uncomputed_tokens() > 0

    def is_finished(self) -> bool:
        """是否完成生成"""
        return all(seq.is_finished() for seq in self.seqs)
```

### 6. SchedulerOutputs
```python
@dataclass
class SchedulerOutputs:
    # 本次调度执行的序列组
    scheduled_seq_groups: List[ScheduledSequenceGroup]

    # token统计
    num_batched_tokens: int             # 批次总token数
    num_prefill_groups: int             # prefill序列组数

    # KV cache内存管理操作
    blocks_to_swap_in: List[Tuple[int, int]]   # CPU→GPU
    blocks_to_swap_out: List[Tuple[int, int]]  # GPU→CPU
    blocks_to_copy: List[Tuple[int, int]]      # GPU内复制

    # 未调度的序列组
    ignored_seq_groups: List[SequenceGroup]
```

### 7. SequenceGroupMetadata
```python
@dataclass
class SequenceGroupMetadata:
    request_id: str                     # 请求ID
    is_prompt: bool                     # 是否是prefill阶段

    seq_data: Dict[int, SequenceData]   # seq_id -> SequenceData
    sampling_params: SamplingParams     # 采样参数
    block_tables: Dict[int, List[int]]  # seq_id -> 物理块表

    token_chunk_size: int               # 本次处理的token数
    # prefill: 可能是全部prompt tokens
    # decode: 通常是1
```

### 8. ExecuteModelRequest
```python
@dataclass
class ExecuteModelRequest:
    seq_group_metadata_list: List[SequenceGroupMetadata]
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_swap_out: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]
    num_lookahead_slots: int = 0
```

### 9. SamplerOutput
```python
@dataclass
class SamplerOutput:
    outputs: List[CompletionSequenceGroupOutput]

    # 可选的性能指标
    sampled_token_ids: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None
```

### 10. CompletionSequenceGroupOutput
```python
@dataclass
class CompletionSequenceGroupOutput:
    samples: List[SequenceOutput]       # 每个序列的采样结果

    # Prompt logprobs (如果请求)
    prompt_logprobs: Optional[PromptLogprobs] = None
```

### 11. SequenceOutput
```python
@dataclass
class SequenceOutput:
    parent_seq_id: int                  # 父序列ID
    output_token: int                   # 采样的token ID
    logprobs: Dict[int, Logprob]        # token_id -> Logprob
```

### 12. RequestOutput
```python
@dataclass
class RequestOutput:
    request_id: str                     # 请求ID
    prompt: str                         # 原始prompt
    prompt_token_ids: List[int]         # prompt的token IDs
    outputs: List[CompletionOutput]     # 生成结果
    finished: bool                      # 是否完成
```

### 13. CompletionOutput
```python
@dataclass
class CompletionOutput:
    index: int                          # 索引
    text: str                           # 生成的文本
    token_ids: List[int]                # 生成的token IDs
    cumulative_logprob: float           # 累积对数概率
    logprobs: Optional[List[Dict[int, Logprob]]]  # 每步的logprobs
    finish_reason: Optional[str]        # 停止原因: "stop"/"length"
```

---

## 总结

### V0架构特点

1. **同步执行**: step()方法同步执行，适合离线批处理
2. **迭代级调度**: 每次step()调度一批请求
3. **高效内存管理**: PagedAttention + 块级KV cache
4. **动态批处理**: 可以在运行时添加/移除请求
5. **Pipeline并行**: 支持多GPU pipeline parallelism

### 性能关键点

1. **调度效率**: Scheduler决定batch组成
2. **内存效率**: KV cache块的分配和回收
3. **计算效率**: CUDA kernel优化(PagedAttention等)
4. **数据传输**: CPU↔GPU数据传输优化

### 适用场景

- **离线批处理**: 大批量文本生成
- **吞吐量优先**: 最大化GPU利用率
- **长序列处理**: 高效的KV cache管理
- **资源受限**: 精细的内存控制

---

## 附录: 关键代码位置

| 组件 | 文件路径 | 行号 |
|------|---------|------|
| LLM.generate | entrypoints/llm.py | 378-475 |
| LLMEngine.add_request | engine/llm_engine.py | 631-730 |
| LLMEngine.step | engine/llm_engine.py | 1212-1440 |
| Scheduler.schedule | core/scheduler.py | - |
| Worker.execute_model | worker/worker.py | 387-449 |
| ModelRunner.execute_model | worker/model_runner.py | - |
| _process_model_outputs | engine/llm_engine.py | 931-1164 |
