# 模块化 AOA 规则

- 适用路径：源码路径均相对 `src/paddlefleet/`：`models/gpt/aoa_generator.py`、`models/gpt/aoa_multi_tower.py`、`models/gpt/aoa_dispatch.py`；`models/gpt/gpt_model.py` 的整模型入口 `GPTModel.gen_aoa_statements` / `gen_inv_aoa_statements`；任何定义或覆写 `gen_aoa_statements` / `gen_inv_aoa_statements` 及其分派钩子（`_gen_qkv_head_aoa_statements`、`_gen_up_gate_fusion_aoa_statements` 等）的模块（当前为 `tensor_parallel/layers.py`、`transformer/mlp.py`、`transformer/attention.py`、`transformer/gated_qkv_aoa.py`、`transformer/multi_latent_attention.py`、`transformer/multi_token_prediction.py`、`transformer/moe/moe_layer.py`、`transformer/moe/moe_expert.py`、`models/gpt/lm_head.py`、`transformers/hyperbody/modeling.py`）；config 字段 `aoa_checkpoint_name_prefix` / `aoa_checkpoint_name_mapping`。测试路径相对仓库根：`tests/single_card_tests/aoa/`
- 触发条件：新增或修改上述方法、钩子或 helper；把某个模型从 `_gen_aoa_config` 迁到模块化路径（删除 `_gen_aoa_config` / `_gen_inv_aoa_config`）；在**已有 AOA 覆写的组件**里新增、重命名、删除参数或 buffer，或改变其 layout / dtype；把子层换成另一种 Linear / Norm / Router 类型；修改 `AOATower` 的划分或 `shares_checkpoint_root`
- 规则来源：Paddle `Layer.gen_aoa_statements` 的基类递归协议与 `paddle.distributed.flex_checkpoint.aoa` 的 DSL 语义；本仓库 `aoa_generator.py` / `aoa_multi_tower.py` / `aoa_dispatch.py` 及各组件覆写的 docstring 声明的契约；PR #2423 修复的两处真实缺陷（`SelfAttention._gen_inv_qkv_head_aoa_statements` 逆向漏转置、`MoELayer` 漏 `latent_norm`）；`tests/single_card_tests/aoa/` 下固化的协议测试
- 引用约定：只用符号名（类名、方法名、helper 名、测试类名）定位代码，不写行号
- 评审范围：只评审本 PR diff 中新增或修改的模块化 AOA 代码。**legacy `_gen_aoa_config` / `_gen_inv_aoa_config`（`transformers/*/modeling.py`、`transformers/aoa_config_base.py`）及其增量改动不在本规则范围内**，不因本文件报告；迁移 PR 删除 legacy 代码时，只评审替代它的模块化实现与分派是否成立

## 机制背景（判定所有规则的前提）

模块化 AOA 的出错形态几乎都是**静默的**，这是本文件所有规则的成因：

1. **同名透传**。基类 `Layer.gen_aoa_statements` 遍历本层自有的 `state_dict(include_sublayers=False)`（参数和持久 buffer 都在内），经 `resolve_names` 得到 `(checkpoint_name, model_name)`，**两者相等时不发语句**，再递归 `_sub_layers`。AOAEngine 对没有任何语句引用的源张量按同名直通。所以"漏一条语句"在名字恰好相同时完全无害，在名字不同时表现为该张量缺失；"本该变换却被当成同名"则表现为直接加载原值。
2. **形状掩盖**。`^T` 漏写在方阵上不改变形状，load 不报错、loss 可能只轻微偏离。PR #2423 的逆向 qkv 漏转置就是被 1280×1280 方阵掩盖的，只在逐张量比对数值时暴露。
3. **语句是有序程序**。宏按优先级展开（`transpose_macro` 为 5，`fused_qkv` / `fused_ffn` 为 6），语句可以原地改写名字，重复一条语句不保证是空操作。因此不能靠事后去重或调换顺序"修"语句集。
4. **覆写即替换**。组件一旦覆写 `gen_aoa_statements`，基类对它自有参数的遍历就不再发生（除非显式调 `super()`，或在覆写里自己对这些参数调 `resolve_names`）。新增一个参数而不改覆写，这个参数就从语句集里消失，回到第 1 条的透传或缺失。
5. **两个方向独立生成**。`gen_whole_model_aoa` 与 `gen_whole_model_inv_aoa` 各自递归，逆向不是正向的反转。一个方向写对不说明另一个方向对。

因此：漏覆盖、漏转置、方向不对称、名字解析绕过 `resolve_*`，这几类问题的现场表现都是"能加载、能训、数不对"。评审必须在静态层面拦住，并要求测试用能暴露它们的方式写。

## 规则优先级

数值错误（漏/多 `^T`、融合布局或 axis 错、dtype 错）≈ 覆盖性缺失（参数或 buffer 未进语句集）> 方向不对称 > 名字解析绕过协议 > PP / 多塔正确性 > 分派与迁移 > 测试缺失 > 注释。

判定严重程度时区分：变换写错或漏写会静默加载出错误数值，属最高优先级；名字恰好相同的漏写由透传兜底、数值正确，若 PR 声称这条语句是必需的才需要追问，否则不报。

## 递归协议与方向独立

- **组件覆写签名固定**为 `(self, ctx, *, structured_name_prefix="", checkpoint_lookup_drop_segment=None)`，两个关键字参数都必须接收，且在委托给子层、`resolve_names` 或 `super()` 时原样传下去。不要改成 `**kwargs` 吞参数，也不要漏传 `checkpoint_lookup_drop_segment`：漏传后 drop-segment 场景（如 MTP 内层）会按错误的名字查 checkpoint，表现为透传或缺失。
- **整模型入口签名故意不同**：`GPTModel.gen_aoa_statements(self, config=None)` 不接收递归参数，容器误递归进来会 `TypeError`，而不是静默产出半份语句。新增整模型入口（含多塔模型）沿用这一约定；不要给它补上 `structured_name_prefix` "让调用更通用"。`test_aoa_modular_generator.py::TestWholeModelAOAEntry::test_recursion_protocol_call_is_rejected` 固化了这点。
- **子层一律委托，不私自重写遍历**。组件处理完自己负责的融合/变换后，其余子层调用 `child.gen_aoa_statements(ctx, structured_name_prefix=f"{prefix}{name}.", ...)`，由子层自己的覆写决定语句。私自用 `named_parameters()` 之类重新遍历子树会绕开子层覆写（例如 Linear 的 `^T`），表现为该子层全部走同名透传。
- **方向独立**：逆向方法必须独立写出，不得通过反转、`replace(" -> ", ...)` 或复用正向语句文本生成；不得给生成函数加 `direction` / `reverse` 一类参数；不得在模块化路径里设 `aoa_config_reverse`（它只属于 legacy 分派的自动推导）。`test_aoa_config_protocol.py::TestDirectionIndependenceLint` 对 `aoa_generator.py` 做了静态检查，新增的生成模块应满足同样的约束。
- **构造路径必须产出覆写类**。覆写只在 live 实例的类型上生效：若模型经 provider / factory / spec 构造子层，确认它真的实例化了带覆写的子类，而不是被同名的基类构造或更外层的 `provide()` 覆盖。需要在已构造实例上换类型时，沿用 `HyperBodyMoELayer` 的 `self.gate.__class__ = ...` 原地换类，不要重建子层（重建会多跑一次 init 并扰动 RNG 序列，同 seed 结果不再一致）。

## 覆盖性：覆写必须管住自己的每个张量

- 组件覆写后，逐个核对该层 `state_dict(include_sublayers=False)` 的每一项（**持久 buffer 也算**，如 router 的 `e_score_correction_bias`）：要么由融合/变换语句覆盖，要么经 `resolve_names` 兜底（名字相同则自动不发），要么委托给子层。覆写类如果自己身上直接挂了参数或持久 buffer（`state_dict(include_sublayers=False)` 非空），覆写里必须逐个处理它们，可照 `MLP.gen_aoa_statements` 的做法：在融合主体之后、委托子层之前，对每个张量调 `resolve_names`，名字不同才发改名语句。像 `SelfAttention` 这样张量全在子层里的覆写，只需把子层逐个委托出去。
- **在已有覆写的组件里新增子层或参数，必须同步改两个方向的覆写**。先例：`MoELayer` 新增 `latent_norm` 后覆写未更新，正逆两向都漏了它（PR #2423 修复，现以 `getattr(self, "latent_norm", None)` 守卫后委托）。评审组件 `__init__` 的 diff 时，若该类有 AOA 覆写，主动检查覆写是否随之修改。
- 可选子层（按配置可能为 None）用 `getattr(..., None) is not None` 守卫后委托，正逆两向守卫条件一致。
- **EP 下按全局 expert 枚举**。`MoELayer` 用 `_global_expert_ctx` 与一个模板 expert 枚举全局 expert id，使每个 rank 产出同一份语句集；新增 expert 相关语句不得只枚举本 rank 持有的 local expert。
- 把子层换成另一种 Linear / Norm / Router 类型时，确认新类型带有与旧类型等价的 AOA 覆写（或确认它本就不需要变换）。例如 `tensor_parallel/layers.py` 的各 Linear 类绑定 `gen_linear_aoa_statements`，换成一个未绑定它的 Linear 会让权重失去 `^T`。

## DSL 语义

- **`^T` 只能写在源侧**。`transpose_macro`（优先级 5）在目标侧遇到 `^T` 会报错；它把 `X^T` 改写成临时张量，再交给 `fused_qkv` / `fused_ffn`（优先级 6）。因此正向写 `q^T, k^T, v^T -> qkv, fused_qkv, ...`；逆向写 `qkv^T -> q, k, v, fused_qkv, ..., axis=0`。融合宏把多个结果放在目标侧时，转置只能拆成单独语句：`MLP` 的逆向先 `fused -> gate, up, fused_ffn`，再分别 `gate^T -> ckpt_gate`、`up^T -> ckpt_up`。
- **两个方向的转置必须成对**。checkpoint 侧是 `[out, in]`、模型侧是 `[in, out]` 的权重，正向和逆向都必须转置。PR #2423 修复的 `SelfAttention._gen_inv_qkv_head_aoa_statements` 原来逆向不转置；评审时对每个带 `^T` 的正向语句，找到它的逆向语句并确认也有 `^T`。
- **bias 与 weight 的融合轴不同**：1D bias 融合要带 `axis=0`（`fused_qkv, ..., axis=0`、`fused_ffn, axis=0`）；routed expert 的 weight 融合由 `moe_expert.py` 覆写成 `axis=1` 普通拼接（EP 下不做 TP 交错）。新增融合时核对轴与该张量的维度、是否 TP 交错。
- **`fused_qkv` 的 head 参数**取自当前层（`num_heads=`、`num_key_value_groups=`），SWA / 异构层按层读取，不要用全局常量。
- **1→1 语句的属性只能是一个算子（如 `permute`、`dtype`），或成对的 `src_dtype` + `dst_dtype`**，后者按 `src_dtype=..., dst_dtype=...` 的顺序书写（AOAEngine 按 `attrs[0]` / `attrs[1]` 的键名分支）。不要只写其中一个。
- **`_ -> B` 表示 B 永不加载、保留初始值**；`A -> _` 表示丢弃源张量 A。前者优先于同名透传，因此只在"checkpoint 确实没有这个张量"时使用。用了 `_ -> B` 的张量，逆向仍必须给出 `B -> <checkpoint 名>`，否则导出的档案会写成模型侧名字，下次加载按 checkpoint 名查不到，再次静默落回初始值。
- **语句列表里的相邻 f-string 漏逗号会被 Python 隐式拼接**成一条非法或含义不同的语句。评审多行语句列表时逐项确认逗号。
- **不得依赖语句幂等**：不要为了"保险"重复发同一条语句，也不要在生成后去重或排序（见机制背景第 3 条）。共享层被 PP 注册两次的情况已由 `_iter_unique_units` 在源头按 `id` 去重。

## 名字解析

- **名字只经 `resolve_names` / `resolve_single_name` / `resolve_checkpoint_name_from_anchor` 得到**，不得手拼 `"model." + local_name`、`f"{ctx.checkpoint_name_prefix}.{...}"` 或硬编码层号。手拼会绕过 `pp_to_single_mapping`（PP 下层号错位）和 `checkpoint_name_mapping`（声明的改名失效），且在默认前缀下恰好正确，单卡测试发现不了。
- **只存在于 checkpoint 侧的名字**（被融合前的 `q_proj` / `k_proj` / `v_proj`、`gate_proj` / `up_proj`）用 `resolve_checkpoint_name_from_anchor` 从模型侧锚点（`qkv_proj.weight`、`up_gate_proj.weight`）推导，不要拿它们去查 `pp_to_single_mapping`（模型里没有这些名字，会 `KeyError`）。
- **`aoa_checkpoint_name_mapping` 的 key 是模型单卡全名**，必须位于模型根前缀之下；不带根的 key 永远匹配不上，`validate_checkpoint_name_mapping` 会拒绝。命中 mapping 的结果即最终名字，不再做前缀替换。新增 mapping 条目时确认它确实需要（名字只差根前缀的情况由前缀替换自动处理）。
- **`aoa_checkpoint_name_prefix = ""` 是合法声明**（checkpoint 位于顶层），只有 `None` 回落默认值 `DEFAULT_CHECKPOINT_NAME_PREFIX`。读取这类字段用 `_resolved` 式的"第一个非 None"，不要写 `getattr(config, ..., None) or default`，后者会把 `""` 吞成默认值。
- 新增的 AOA 标识符沿用 `checkpoint_*` / `model_*` 词汇，不用 `hf_*` / `fleet_*` 指代两侧（提示级）。

## dtype

- **cast 只放在模型侧的最小子类里**，不放进共享组件（Linear、`MLP`、`SelfAttention`、`MoELayer` 的基类覆写）。共享组件服务所有模型，一处 cast 会改变所有模型的加载数值。先例：`transformers/hyperbody/modeling.py` 的 `HyperBodyGateRouter`、`HyperBodyMhcContract`、`HyperBodyMTPLayer` 只在本模型内覆写，附加 cast 语句。
- **dtype 值取自 live 张量或显式的档案契约常量**：模型侧用 `_dtype_attr(self.weight)` 一类从参数读出；checkpoint 侧 dtype 用类常量声明（如 `_AOA_GATE_CHECKPOINT_DTYPE`），并在 docstring 写明依据。不要按注释、配置名或经验硬写 `'bfloat16'` / `'float32'`，档案实际 dtype 与假设不符时会把 FP32 张量导出成 BF16。
- 两个方向分别处理：正向可用 `dtype=`；逆向若档案 dtype 与模型不同，用 `src_dtype` + `dst_dtype`；档案 dtype 与模型一致时逆向不需要 cast（`HyperBodyMhcContract` 的 docstring 说明了这种情况）。

## PP 与多塔

- 生成器不得读 rank 本地状态或 TP 分片后的局部形状来决定语句：`globalize_statements` 的正确性依赖"TP / DP / sharding 各 rank 产出相同语句、PP 各 stage 拥有互不相交的层"。读局部 shape 决定 head 数或 split 点会让各 rank 语句不一致。
- `pipeline_model_parallel_size > 1` 但没有 PP group 时，`globalize_statements` 抛 `RuntimeError`，这是有意的硬失败，不要改成降级为 rank 本地语句。多塔路径用 `model.config`（整模型 config）调用它，不要换成某个塔的 view config：塔的 view 可能不带 `pipeline_model_parallel_size`，会让这道检查失效。
- **`AOATower` 划分必须让每个 live unit 恰好属于一个塔**，`_iter_tower_units` 对 0 个或多个 owner 都会抛错；修改 `owns_unit` 时检查边界层（共享层、MTP、空层）。
- **多个目的端读同一个 checkpoint 根时，相关塔必须设 `shares_checkpoint_root=True`**。否则 `_copy_path_for_shared_checkpoint_roots` 不会把直读改写成经临时张量的拷贝，同一源张量只会加载到其中一端，另一端保持初始值或未初始化的 `paddle.empty`（数值上可能是 `1e-37` 量级的垃圾值），不报错。
- 塔级 checkpoint 覆写（`checkpoint_name_prefix` / `checkpoint_name_mapping`）只给"读的 checkpoint 区域与自身命名不同"的塔，例如 MTP warm-start；名字本来就对的塔不传。

## 分派与迁移

- **分派是 legacy 优先**：`resolve_aoa_config` 先看 `hasattr(model, "_gen_aoa_config")`，只有没有它时才走模块化入口；`resolve_inv_aoa_config` 还会从 `_gen_aoa_config` 自动推导逆向。因此迁移 PR 必须从模型类（及其父类、外部挂上去的 lambda / `MethodType`）中**删干净** `_gen_aoa_config` 与 `_gen_inv_aoa_config`，否则模块化代码永远不会被调用，组件测试全绿但生产仍走 legacy。评审时要求给出证据（测试断言分派结果，或说明 `hasattr` 为 False 的依据）。
- 非 `GPTModel` 子类的模型（如 `HyperBodyUnifiedModel`）要自己覆写两个整模型入口；只覆写一个方向会让另一方向在分派时 `RuntimeError`。
- **迁移的等价性证据要看数值与档案 schema，不看语句条数和 loss**。模块化路径按需生成（同名无变换不发语句），条数天然与 legacy 不同；loss 对齐也可能掩盖导出 key 集合或 dtype 的差异。要求的证据是：同一份 checkpoint 经两条路径加载后逐张量相等（覆盖非方阵权重），以及导出档案的 key 集合与 dtype 与 legacy 一致。

## 已接受的缺口（不要报告）

以下是现有设计有意不做或已知未覆盖的点，评审时不要据此提意见，除非本 PR 声称要修它们：

- AOAEngine 不检查多条语句写同一目标；生成器侧也没有统一的重复目标断言，不要求本 PR 补一个。
- 导出方向不校验目标 key 集合是否完整。
- 逆向没有对应的 copy-path 改写：`_copy_path_for_shared_checkpoint_roots` 只作用于加载方向。
- 模块化语句条数与 legacy 不同（按需生成），不是回归。

## 测试要求

新增或修改模块化 AOA 代码必须补测试，统一放在 `tests/single_card_tests/aoa/`（`ci/single_card_test.sh` 只收集 `tests/single_card_tests/` 下的文件，且每个文件单独起进程跑）。必须覆盖的点：

- **两个方向都要测**，且逆向断言独立写出，不要用正向结果反推期望值。
- **数值往返用非方阵形状**。字符串断言只能证明语句长什么样，证明不了转置和融合布局对。凡涉及 `^T` 或融合宏的改动，至少一条测试用真实张量走 `dist.save_state_dict` / `dist.load_state_dict`（`aoa_config={"aoa_statements": ...}`），在 `tempfile.TemporaryDirectory` 内完成，形状选成各维互不相等。参考 `test_aoa_self_attention_component.py::TestSelfAttentionQKVRoundTrip`（hidden=6、q_out=32、kv_out=8、fused_out=48，正向后逆向还原原 checkpoint），以及多塔 fan-out 的 `test_aoa_multi_tower.py::TestFannedOutLoadReachesBothEnds`。方阵测试对漏转置无效，按测试缺失报告。
- **覆盖性用结构断言**：断言"某张量出现在语句集里"或"没有任何语句提及它"，不要只断言语句条数。按需生成下条数会因同名透传而变化，条数断言既可能误报也可能空过。新增子层时补一条"两个方向都提及该子层"的断言，参考 `test_aoa_moe_layer_component.py::TestMoELayerLatentNorm`。
- **反向用例**：可选子层为 None 或可选 bias 关闭时，断言对应语句不出现（参考 `test_aoa_self_attention_component.py` 的 `test_no_qkv_bias_statement_when_absent`）。
- **协议性测试随结构变化同步更新**：新增覆写类时补"该类 `__dict__` 中有这些覆写"的存在性断言（参考 `TestSelfAttentionClassContract`）；改递归看 `test_aoa_modular_generator.py::TestRecursionProtocol`，改整模型入口看同文件的 `TestWholeModelAOAEntry`，改 PP 汇总看 `TestGatherGuardedOnLivePipelineGroup`；改分派看 `test_aoa_dispatch.py`；改生成模块看 `test_aoa_config_protocol.py::TestDirectionIndependenceLint` 是否仍覆盖。
- 真实组件需要 TP 进程组时，沿用既有做法：把被测覆写方法绑定到一个可控的 `paddle.nn.Layer` 替身上（如 `test_aoa_self_attention_component.py` 的 `_make_attn_leaf`），替身持有与真实组件同名的子层与属性。替身与生产组件的子层类型不同时，测试里应说明两者的参数集合一致。
- 往返测试要设置并在 `tearDown` 中恢复 `PADDLE_TRAINER_*` 等环境变量与 device，参考 `TestSelfAttentionQKVRoundTrip.setUp` / `tearDown`。

只改了 AOA 语句而没有任何测试变更的 PR，必须报告测试缺失，并指出缺的是上面哪一类。

## 注释与可追溯性

- 每个覆写的 docstring 写清该组件发哪些语句、为什么需要变换（布局 `[out, in]` 与 `[in, out]` 的对应、为什么融合轴是这个值、为什么某个张量用 `_ ->`）。既有约定见 `MLP._gen_inv_up_gate_fusion_aoa_statements`（解释为什么转置要拆成单独语句）和 `GPTModel.gen_aoa_statements`（解释签名为何不接收递归参数）。非显然的变换缺少说明时报告。
- 档案 dtype 常量、`_ -> B`、`shares_checkpoint_root=True` 这类依赖外部档案事实的选择，注释要写明依据的是档案的什么事实，而不是只写结论。
- 代码注释不引用仓库外的设计文档编号或章节号，读者无法打开它们。
