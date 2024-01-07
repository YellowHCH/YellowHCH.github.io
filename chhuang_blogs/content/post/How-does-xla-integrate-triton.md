---
title: How does xla integrate with triton
tags: ["Triton", "XLA", "Framework"]
date: 2023-12-24

---
# Brief

xla的codegen可以选择使用triton backend 对少量的op进行codegen，包括`matmul`和`softmax`。

通过选项`xla_gpu_enable_triton_gemm`和`xla_gpu_enable_triton_softmax_fusion`控制是否开启。

# matmul

`GemmRewriterTriton`实现了选择 gemm 走 triton 后端的逻辑。

- Rewriter

```c++
// Extracts into fused computations parts of HLO graph including dot()
// operations that can target the triton GEMM emitter.
class GemmRewriterTritonVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterTritonVisitor(const GpuVersion gpu_version)
      : gpu_version_(gpu_version) {}
  // NOTE. 分析gemm是否走triton 后端
  Status HandleDot(HloInstruction* dot) override {
    VLOG(5) << dot->ToString();
    // NOTE. 检查数据类型是否支持，检查是否有fuse inputs outputs的机会。
    if (!IsTritonHandledGEMM(*dot, gpu_version_)) {
      return OkStatus();
    }

    // NOTE. ...

    HloComputation* computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                            /*is_entry=*/false);
    HloInstruction* dot_fusion =
        dot->parent()->AddInstruction(HloInstruction::CreateFusion(
            computation->root_instruction()->shape(),
            HloInstruction::FusionKind::kCustom, call_operands, computation));
    dot_fusion->GetModule()->SetAndUniquifyInstrName(dot_fusion,
                                                     suggested_name);

    TF_ASSIGN_OR_RETURN(auto backend_config,
                        dot_fusion->backend_config<FusionBackendConfig>());
    // NOTE. 将后端设置成triton，使用triton emit
    backend_config.set_kind(std::string(kTritonGemmFusionKind));
    TF_RETURN_IF_ERROR(dot_fusion->set_backend_config(backend_config));
    // NOTE. ...
  }

 private:
  GpuVersion gpu_version_;
};
```

- triton gemm emitter

```c++
StatusOr<LaunchDimensions> TritonWrapper(
    absl::string_view fn_name, const HloComputation* hlo_computation,
    absl::string_view fusion_kind, const se::CudaComputeCapability& cc,
    const GpuDeviceInfo& device_info,
    const AutotuneResult::TritonGemmKey& config, llvm::Module* llvm_module,
    LaunchDimensionsGenerator generator, mlir::MLIRContext& mlir_context) {

  // NOTE. 生成 triton dialect
  mlir_context.loadDialect<mt::TritonDialect>();
  mlir::OpBuilder b(&mlir_context);
  auto loc = mlir::NameLoc::get(b.getStringAttr(hlo_computation->name()));
  auto triton_module = mlir::ModuleOp::create(loc);
  b.setInsertionPointToEnd(triton_module.getBody());

  // Build Triton kernel.
  // NOTE. 构造函数入参
  SmallVector<Type> fn_arg_types;
  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    fn_arg_types.push_back(mt::PointerType::get(
        TritonType(b, p->shape().element_type()), mn::kGlobalMemorySpace));
  }

  for (const ShapeUtil::IndexedShape& s :
       ShapeUtil::GetLeafShapes(hlo_computation->root_instruction()->shape())) {
    fn_arg_types.push_back(mt::PointerType::get(
        TritonType(b, s.shape.element_type()), mn::kGlobalMemorySpace));
  }

  // NOTE. 生成函数体
  auto fn = b.create<mt::FuncOp>(loc, fn_name,
                                 b.getFunctionType(fn_arg_types, std::nullopt));
  for (int i = 0; i < fn.getNumArguments(); ++i) {
    fn.setArgAttr(i, "tt.divisibility", b.getIntegerAttr(b.getI32Type(), 16));
  }
  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());
  // NOTE. triton codegen 依赖libdevice中的 math functions.
  const std::string libdevice_path =
      nvptx::LibDevicePath(hlo_computation->parent()
                               ->config()
                               .debug_options()
                               .xla_gpu_cuda_data_dir());
  // NOTE. 调用generator填充函数body
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      generator(b, libdevice_path, hlo_computation, fn, config,
                                device_info.shared_memory_per_block_optin));

  b.create<mt::ReturnOp>(loc);

  // NOTE. 基本上是把triton compile的py实现用cpp实现。
  // Compile Triton kernel to LLVM.
  mlir::PassManager pm(&mlir_context);

  std::optional<llvm::raw_fd_ostream> log_stream;
  const HloModule* hlo_module = hlo_computation->parent();
  // NOTE. 调用triton的pipeline编译ttir
  CreateTritonPipeline(pm, cc, config.num_warps(), config.num_stages());
  // Triton generates pointers to the global address space, while XLA needs a
  // kernel signature with pointers to the generic address space.
  pm.addPass(std::make_unique<GeneralizeKernelSignaturePass>());
  // llvm::Linker::linkModules() segfaults if we don't strip locations.
  pm.addPass(mlir::createStripDebugInfoPass());

  bool succeeded = mlir::succeeded(pm.run(triton_module));

  const int shared_mem_bytes =
      triton_module->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared")
          .getInt();
  VLOG(2) << "Shared memory usage: " << shared_mem_bytes << " B";
  if (shared_mem_bytes > device_info.shared_memory_per_block_optin) {
    return ResourceExhausted("Shared memory size limit exceeded.");
  }
  launch_dimensions.SetSharedMemBytes(shared_mem_bytes);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::Module> ll_triton_module,
                      TranslateLLVMToLLVMIR(&llvm_module->getContext(),
                                            triton_module, libdevice_path));
  LogAndVerify(ll_triton_module.get());

  // Integrate LLVM matmul kernel into XLA's LLVM module.
  ll_triton_module->eraseNamedMDNode(
      ll_triton_module->getNamedMetadata("nvvm.annotations"));
  ll_triton_module->setDataLayout(llvm_module->getDataLayout());
  // Use override flag because libdevice functions can be present in both.
  CHECK(!llvm::Linker::linkModules(*llvm_module, std::move(ll_triton_module),
                                   llvm::Linker::Flags::OverrideFromSrc));
  LogAndVerify(llvm_module);

  return launch_dimensions;
}
```

- generator

```c++
using LaunchDimensionsGenerator = std::function<StatusOr<LaunchDimensions>(
    mlir::OpBuilder, absl::string_view, const HloComputation*,
    mlir::triton::FuncOp, const AutotuneResult::TritonGemmKey&, int)>;
```

- matmul impl

```c++
StatusOr<LaunchDimensions> MatMul(
    mlir::OpBuilder builder, absl::string_view libdevice_path,
    const HloComputation* computation, mlir::triton::FuncOp fn,
    const tensorflow::AutotuneResult::TritonGemmKey& config, int shmem_budget) {
  const HloDotInstruction* dot_instr = DynCast<HloDotInstruction>(
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot));
  // Use 32-bit indexing if addressing any of the inputs or the output (which
  // could grow if split_k is set) does not cross the INT_MAX boundary.
  // Otherwise, fall back to 64-bit indexing, which is slower.
  bool use_64bit_indexing =
      ShapeUtil::ElementsIn(dot_instr->operand(0)->shape()) > INT_MAX ||
      ShapeUtil::ElementsIn(dot_instr->operand(1)->shape()) > INT_MAX ||
      ShapeUtil::ElementsIn(dot_instr->shape()) * config.split_k() > INT_MAX;
  if (use_64bit_indexing) {
    return MatMulImpl<int64_t>(builder, libdevice_path, dot_instr, fn, config,
                               shmem_budget);
  } else {
    return MatMulImpl<int32_t>(builder, libdevice_path, dot_instr, fn, config,
                               shmem_budget);
  }
}
```
- matmul ttir impl

先生成ttir然后照着用builder实现。应该是因为xla的codegen是在c++环境的，无法复用triton的前端，所以要自己实现。

```c++
template <typename IndexT>
StatusOr<LaunchDimensions> MatMulImpl(
    mlir::OpBuilder builder, absl::string_view libdevice_path,
    const HloDotInstruction* dot_instr, mlir::triton::FuncOp fn,
    const tensorflow::AutotuneResult::TritonGemmKey& config, int shmem_budget) {
  const HloInstruction* root = dot_instr->parent()->root_instruction();
  CHECK(!root->shape().IsTuple());

  // We'll be creating a lot of instructions from a single dot, use an
  // implicit loc builder so we don't have to pass around the location all the
  // time.
  auto loc = mlir::NameLoc::get(builder.getStringAttr(dot_instr->name()));
  mlir::ImplicitLocOpBuilder b(loc, builder);
  Type i32_ty = b.getI32Type();
  Type int_ty;
  if constexpr (std::is_same_v<IndexT, int64_t>) {
    int_ty = b.getI64Type();
  } else {
    int_ty = b.getI32Type();
  }
  const DotDimensionNumbers& dims = dot_instr->dot_dimension_numbers();
  const DotFusionAnalysis analysis(dot_instr->parent(), config.split_k());

  // Rely on dot decomposer: there is just one contracting and one
  // non-contracting dimension on each side + batch ones optionally.
  CHECK_EQ(dims.lhs_contracting_dimensions_size(), 1);
  CHECK_EQ(dims.rhs_contracting_dimensions_size(), 1);

  const bool have_split_k = config.split_k() > 1;
  if (have_split_k) {
    // Split-K dimension has to be the first batch one and have an index
    // just before the contracting one.
    // Size of this dimension has to match the split_k value.
    CHECK_EQ(dims.lhs_batch_dimensions(0),
             dims.lhs_contracting_dimensions(0) - 1);
    CHECK_EQ(dims.rhs_batch_dimensions(0),
             dims.rhs_contracting_dimensions(0) - 1);
    CHECK_EQ(config.split_k(), dot_instr->operand(0)->shape().dimensions(
                                   dims.lhs_contracting_dimensions(0) - 1));
    CHECK_EQ(config.split_k(), dot_instr->operand(1)->shape().dimensions(
                                   dims.rhs_contracting_dimensions(0) - 1));
  }

  CHECK_LE(dims.lhs_batch_dimensions_size(), 1 + have_split_k);
  const bool have_batch = dims.lhs_batch_dimensions_size() - have_split_k;
  CHECK_EQ(dot_instr->operand(0)->shape().rank(),
           2 + have_split_k + have_batch);
  const int64_t lhs_noncontracting_dim_idx =
      GetNonContractingDims(dot_instr->operand(0)->shape(),
                            dims.lhs_batch_dimensions(),
                            dims.lhs_contracting_dimensions())
          .value()[0];
  const int64_t rhs_noncontracting_dim_idx =
      GetNonContractingDims(dot_instr->operand(1)->shape(),
                            dims.rhs_batch_dimensions(),
                            dims.rhs_contracting_dimensions())
          .value()[0];

  // Logical output dimensions are always ordered as:
  //   split-K, batch, non-contracting LHS, non-contracting RHS,
  // where split-K and batch are optional.
  const int rhs_nc_out_idx = dot_instr->shape().rank() - 1;
  const int lhs_nc_out_idx = dot_instr->shape().rank() - 2;
  const int split_k_out_idx = have_split_k ? 0 : -1;
  const int batch_out_idx = have_batch ? (have_split_k ? 1 : 0) : -1;

  // LHS non-contracting dimension length.
  // LHS non-contracting can be split, so this holds its full size unlike the
  // m_minor.
  int m =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, lhs_nc_out_idx)
          ->at(0)
          .count;

  // Contracting dimension length.
  const int k = dot_instr->operand(0)->shape().dimensions(
                    dims.lhs_contracting_dimensions(0)) *
                config.split_k();

  // For now all parameters of one scope (dot LHS, RHS) are required to have the
  // same physical layout = use the same indices in tiles. This is enforced by
  // construction in the Triton GEMM rewriter.

  // LHS non-contracting can be split into two.
  bool lhs_nc_split = false;
  // Either batch size or upper part of the length of a split nc dimension.
  int batch_size = 1;
  IndexT stride_lhs_m = 0;
  IndexT stride_lhs_k = 0;
  IndexT stride_lhs_batch = 0;
  IndexT stride_rhs_batch = 0;
  if (!analysis.ScopeParameters(DotFusionAnalysis::Scope::LHS).empty()) {
    const HloInstruction* lhs_param0 =
        *analysis.ScopeParameters(DotFusionAnalysis::Scope::LHS).begin();
    const DotFusionAnalysis::DimIterationSpec* lhs_nc_iter_spec =
        analysis.IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                          lhs_noncontracting_dim_idx);
    lhs_nc_split = lhs_nc_iter_spec->size() > 1;
    // For now split non-contracting and batch are not supported simultaneously
    // because they are implemented via same mechanism.
    CHECK_LE(have_batch + lhs_nc_split, 1);
    if (lhs_nc_split) {
      batch_size = lhs_nc_iter_spec->at(1).count;
      CHECK_GE(batch_size, 1);
      stride_lhs_batch = lhs_nc_iter_spec->at(1).stride;
      CHECK_GE(stride_lhs_batch, 1);
    } else if (have_batch) {
      const int64_t lhs_batch_dim_idx =
          *(dims.lhs_batch_dimensions().cend() - 1);
      batch_size = analysis
                       .IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                                 lhs_batch_dim_idx)
                       ->at(0)
                       .count;
      CHECK_GE(batch_size, 1);
      stride_lhs_batch = analysis
                             .IterSpec(DotFusionAnalysis::Scope::LHS,
                                       lhs_param0, lhs_batch_dim_idx)
                             ->at(0)
                             .stride;
      CHECK_GE(stride_lhs_batch, 1);
    }

    CHECK_EQ(lhs_nc_iter_spec->size(), 1 + lhs_nc_split);
    CHECK_EQ(analysis
                 .IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                           dims.lhs_contracting_dimensions(0))
                 ->size(),
             1);
    stride_lhs_m = lhs_nc_iter_spec->at(0).stride;
    stride_lhs_k = analysis
                       .IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                                 dims.lhs_contracting_dimensions(0))
                       ->at(0)
                       .stride;
    // Just the fastest-varying part of it if the dimension is split.
    m = lhs_nc_iter_spec->at(0).count;
  }

  CHECK_GE(m, 1);

  IndexT stride_rhs_k = 0;
  IndexT stride_rhs_n = 0;
  if (!analysis.ScopeParameters(DotFusionAnalysis::Scope::RHS).empty()) {
    const HloInstruction* rhs_param0 =
        *analysis.ScopeParameters(DotFusionAnalysis::Scope::RHS).begin();
    // Splitting of RHS non-contracting is not supported yet.
    CHECK_EQ(analysis
                 .IterSpec(DotFusionAnalysis::Scope::RHS, rhs_param0,
                           rhs_noncontracting_dim_idx)
                 ->size(),
             1);
    stride_rhs_k = analysis
                       .IterSpec(DotFusionAnalysis::Scope::RHS, rhs_param0,
                                 dims.rhs_contracting_dimensions(0))
                       ->at(0)
                       .stride;
    stride_rhs_n = analysis
                       .IterSpec(DotFusionAnalysis::Scope::RHS, rhs_param0,
                                 rhs_noncontracting_dim_idx)
                       ->at(0)
                       .stride;
    if (have_batch) {
      const int64_t rhs_batch_dim_idx =
          *(dims.rhs_batch_dimensions().cend() - 1);
      stride_rhs_batch = analysis
                             .IterSpec(DotFusionAnalysis::Scope::RHS,
                                       rhs_param0, rhs_batch_dim_idx)
                             ->at(0)
                             .stride;
      CHECK_GE(stride_rhs_batch, 1);
    }
  }

  constexpr int group_m = 8;

  IndexT stride_out_m =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, lhs_nc_out_idx)
          ->at(0)
          .stride;
  const int64_t n =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, rhs_nc_out_idx)
          ->at(0)
          .count;
  CHECK_GE(n, 1);
  IndexT stride_out_n =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, rhs_nc_out_idx)
          ->at(0)
          .stride;
  IndexT stride_out_split_k = 0;
  if (have_split_k) {
    stride_out_split_k =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, split_k_out_idx)
            ->at(0)
            .stride;
    CHECK_GE(stride_out_split_k, 1);
  }
  IndexT stride_out_batch = 0;
  if (have_batch) {
    stride_out_batch =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, batch_out_idx)
            ->at(0)
            .stride;
    CHECK_GE(stride_out_batch, 1);
  } else if (lhs_nc_split) {
    // Dimension of the output produced by the non-contracting LHS one
    // is physically contiguous even if the producing LHS one is split.
    // Because the major part of the split is implemented using the batch
    // logic stride_out_batch is populated here as the stride of the minor
    // part times its size.
    stride_out_batch = stride_out_m * m;
  }

  const int block_m = config.block_m();
  const int block_k = config.block_k();
  const int block_n = config.block_n();

  CHECK_GE(block_m, 16);
  CHECK_GE(block_k, 16);
  CHECK_GE(block_n, 16);

  const int grid_m = ceil(1.0 * m / block_m);
  const int grid_n = ceil(1.0 * n / block_n);
  const int width = group_m * grid_n;

  Type dot_output_ty = TritonType(b, dot_instr->shape().element_type());

  {
    int required_shmem_size = 0;
    for (const HloInstruction* hlo :
         analysis.ScopeParameters(DotFusionAnalysis::Scope::LHS)) {
      required_shmem_size += block_m * ShapeUtil::ByteSizeOfPrimitiveType(
                                           hlo->shape().element_type());
    }
    for (const HloInstruction* hlo :
         analysis.ScopeParameters(DotFusionAnalysis::Scope::RHS)) {
      required_shmem_size += block_n * ShapeUtil::ByteSizeOfPrimitiveType(
                                           hlo->shape().element_type());
    }
    required_shmem_size *= block_k * config.num_stages();
    if (required_shmem_size > shmem_budget) {
      return ResourceExhausted("Requires too much shared memory: %d > %d",
                               required_shmem_size, shmem_budget);
    }
  }

  // Data type of dot() immediate inputs.
  Type dot_input_ty = b.getF32Type();
  {
    const Type lhs_ty =
        TritonType(b, dot_instr->operand(0)->shape().element_type());
    const Type rhs_ty =
        TritonType(b, dot_instr->operand(1)->shape().element_type());
    CHECK(lhs_ty == rhs_ty);
    dot_input_ty = lhs_ty;
  }
  // TODO(b/266862493): Accumulator can be integer too.
  // Otherwise only f64 x f64 -> f64 uses f64 accumulator.
  mlir::FloatType acc_ty = (dot_output_ty.isF64() && dot_input_ty.isF64())
                               ? b.getF64Type()
                               : b.getF32Type();

  // X block size is 32-bit, Y and Z are 16-bit. Use X for large dimensions.
  constexpr int64_t kBlockCountYZLimit = 65536;
  const bool large_batch = batch_size >= kBlockCountYZLimit;
  auto pid_batch = b.create<mt::GetProgramIdOp>(
      large_batch ? mt::ProgramIDDim::X : mt::ProgramIDDim::Y);
  auto pid_nc = b.create<mt::GetProgramIdOp>(large_batch ? mt::ProgramIDDim::Y
                                                         : mt::ProgramIDDim::X);
  auto pid_k = b.create<mt::GetProgramIdOp>(mt::ProgramIDDim::Z);

  // In the imaginary situation where both batch size and grid_m * grid_n
  // are over 65535 we have to give up. Given the minimal m, n block sizes of 16
  // this requires at least 256 GB of output.
  CHECK_LT(batch_size * grid_m * grid_n,
           kBlockCountYZLimit * kBlockCountYZLimit);

  const LaunchDimensions launch_dimensions{
      {large_batch ? batch_size : grid_m * grid_n,
       large_batch ? grid_m * grid_n : batch_size, config.split_k()},
      {config.num_warps() * WarpSize(), 1, 1}};

  auto group_id = b.create<ma::DivSIOp>(pid_nc, CreateConst(b, i32_ty, width));
  ma::ConstantOp group_m_op = CreateConst(b, i32_ty, group_m);
  auto first_pid_m = b.create<ma::MulIOp>(group_id, group_m_op);
  auto sub0 = b.create<ma::SubIOp>(CreateConst(b, i32_ty, grid_m), first_pid_m);
  auto group_size = b.create<ma::SelectOp>(
      b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, sub0, group_m_op), sub0,
      group_m_op);

  // Extend int32 indexes to int64, if necessary.
  auto convert_scalar = [&](Value value) -> Value {
    if constexpr (std::is_same_v<IndexT, int64_t>) {
      return b.create<ma::ExtSIOp>(int_ty, value);
    }
    return value;
  };
  auto convert_range = [&](Value value) -> Value {
    if constexpr (std::is_same_v<IndexT, int64_t>) {
      auto type = mlir::RankedTensorType::get(
          value.dyn_cast<TensorValue>().getType().getShape(), int_ty);
      return b.create<ma::ExtSIOp>(type, value);
    }
    return value;
  };

  auto pid_m = b.create<ma::AddIOp>(first_pid_m,
                                    b.create<ma::RemSIOp>(pid_nc, group_size));
  auto pid_m_stride =
      b.create<ma::MulIOp>(pid_m, CreateConst(b, i32_ty, block_m));
  // TODO(b/270351731): Consider regenerating range_m to reduce register
  // pressure if we figure out how to make this optimization survive CSE.
  auto range_m =
      b.create<ma::AddIOp>(Splat(b, pid_m_stride, block_m), Range(b, block_m));

  auto pid_n = b.create<ma::DivSIOp>(
      b.create<ma::RemSIOp>(pid_nc, CreateConst(b, i32_ty, width)), group_size);
  auto pid_n_stride =
      b.create<ma::MulIOp>(pid_n, CreateConst(b, i32_ty, block_n));
  auto range_n =
      b.create<ma::AddIOp>(Splat(b, pid_n_stride, block_n), Range(b, block_n));

  auto range_k = b.create<ma::AddIOp>(
      Splat(b, b.create<ma::MulIOp>(pid_k, CreateConst(b, i32_ty, block_k)),
            block_k),
      Range(b, block_k));

  SmallVector<int64_t, 2> shape_m_1{block_m, 1};
  auto range_lhs_m = convert_range(
      b.create<ma::RemSIOp>(range_m, CreateConst(b, i32_ty, m, block_m)));
  auto lhs_offsets_m =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_lhs_m, 1),
                           CreateConst(b, int_ty, stride_lhs_m, shape_m_1));
  SmallVector<int64_t, 2> shape_1_k{1, block_k};
  auto lhs_offsets_k = b.create<ma::MulIOp>(
      b.create<mt::ExpandDimsOp>(convert_range(range_k), 0),
      CreateConst(b, int_ty, stride_lhs_k, shape_1_k));
  SmallVector<int64_t, 2> shape_m_k{block_m, block_k};
  auto lhs_offset_batch = b.create<ma::MulIOp>(
      convert_scalar(pid_batch), CreateConst(b, int_ty, stride_lhs_batch));
  auto lhs_offsets_init = b.create<ma::AddIOp>(
      Broadcast(b, lhs_offsets_m.getResult().template cast<TensorValue>(),
                shape_m_k),
      Broadcast(b, lhs_offsets_k.getResult().template cast<TensorValue>(),
                shape_m_k));
  lhs_offsets_init = b.create<ma::AddIOp>(
      lhs_offsets_init, Splat(b, lhs_offset_batch, shape_m_k));

  SmallVector<int64_t, 2> shape_k_1{block_k, 1};
  auto rhs_offsets_k = b.create<ma::MulIOp>(
      b.create<mt::ExpandDimsOp>(convert_range(range_k), 1),
      CreateConst(b, int_ty, stride_rhs_k, shape_k_1));
  SmallVector<int64_t, 2> shape_1_n{1, block_n};
  auto range_rhs_n = convert_range(
      b.create<ma::RemSIOp>(range_n, CreateConst(b, i32_ty, n, block_n)));
  auto rhs_offsets_n =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_rhs_n, 0),
                           CreateConst(b, int_ty, stride_rhs_n, shape_1_n));
  SmallVector<int64_t, 2> shape_k_n{block_k, block_n};
  auto rhs_offset_batch = b.create<ma::MulIOp>(
      convert_scalar(pid_batch), CreateConst(b, int_ty, stride_rhs_batch));
  auto rhs_offsets_init = b.create<ma::AddIOp>(
      Broadcast(b, rhs_offsets_k.getResult().template cast<TensorValue>(),
                shape_k_n),
      Broadcast(b, rhs_offsets_n.getResult().template cast<TensorValue>(),
                shape_k_n));
  rhs_offsets_init = b.create<ma::AddIOp>(
      rhs_offsets_init, Splat(b, rhs_offset_batch, shape_k_n));
  SmallVector<int64_t, 2> shape_m_n{block_m, block_n};
  ma::ConstantOp accumulator_init = CreateConst(b, acc_ty, 0, shape_m_n);

  auto body_builder = [&](mlir::OpBuilder&, mlir::Location, Value ki,
                          mlir::ValueRange iterArgs) {
    Value lhs_offsets = iterArgs[0];
    Value rhs_offsets = iterArgs[1];
    Value accumulator = iterArgs[2];
    Value lhs_mask = nullptr;
    Value rhs_mask = nullptr;
    // TODO(b/269726484): Peel the loop instead of inserting a masked load in
    // every iteration, even the ones that do not need it.
    const bool need_masking = k % (block_k * config.split_k()) > 0;
    if (need_masking) {
      auto elements_in_tile =
          b.create<ma::SubIOp>(CreateConst(b, i32_ty, k), ki);
      lhs_mask =
          Broadcast(b,
                    b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                         b.create<mt::ExpandDimsOp>(range_k, 0),
                                         Splat(b, elements_in_tile, shape_1_k))
                        .getResult()
                        .template cast<TensorValue>(),
                    shape_m_k);
      rhs_mask =
          Broadcast(b,
                    b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                         b.create<mt::ExpandDimsOp>(range_k, 1),
                                         Splat(b, elements_in_tile, shape_k_1))
                        .getResult()
                        .template cast<TensorValue>(),
                    shape_k_n);
    }

    // For now use one shape for LHS inputs and one for RHS.
    absl::flat_hash_map<const HloInstruction*, Value> values_lhs;
    Value dot_input_lhs =
        EmitScope(b, libdevice_path, fn,
                  dot_instr->parent()->MakeInstructionPostOrderFrom(
                      const_cast<HloInstruction&>(*dot_instr->operand(0))),
                  values_lhs, lhs_offsets, lhs_mask);
    absl::flat_hash_map<const HloInstruction*, Value> values_rhs;
    Value dot_input_rhs =
        EmitScope(b, libdevice_path, fn,
                  dot_instr->parent()->MakeInstructionPostOrderFrom(
                      const_cast<HloInstruction&>(*dot_instr->operand(1))),
                  values_rhs, rhs_offsets, rhs_mask);

    if (need_masking) {
      dot_input_lhs = b.create<ma::SelectOp>(lhs_mask, dot_input_lhs,
                                             ZerosLike(b, dot_input_lhs));
      dot_input_rhs = b.create<ma::SelectOp>(rhs_mask, dot_input_rhs,
                                             ZerosLike(b, dot_input_rhs));
    }

    auto accumulator_next = b.create<mt::DotOp>(
        dot_input_lhs, dot_input_rhs, accumulator,
        /*allowTF32=*/tsl::tensor_float_32_execution_enabled());

    Value lhs_offsets_next = b.create<ma::AddIOp>(
        lhs_offsets,
        CreateConst(b, int_ty, block_k * config.split_k() * stride_lhs_k,
                    shape_m_k));
    Value rhs_offsets_next = b.create<ma::AddIOp>(
        rhs_offsets,
        CreateConst(b, int_ty, block_k * config.split_k() * stride_rhs_k,
                    shape_k_n));

    b.create<mlir::scf::YieldOp>(
        mlir::ValueRange{lhs_offsets_next, rhs_offsets_next, accumulator_next});
  };
  Value acc_final =
      b.create<mlir::scf::ForOp>(
           /*lowerBound=*/b.create<ma::ConstantIntOp>(0, /*width=*/32),
           /*upperBound=*/b.create<ma::ConstantIntOp>(k, /*width=*/32),
           /*step=*/
           b.create<ma::ConstantIntOp>(block_k * config.split_k(),
                                       /*width=*/32),
           /*iterArgs=*/
           mlir::ValueRange{lhs_offsets_init, rhs_offsets_init,
                            accumulator_init},
           body_builder)
          .getResult(2);
  absl::flat_hash_map<const HloInstruction*, Value> values_out;
  values_out[dot_instr] =
      Cast(b, acc_final, TritonType(b, dot_instr->shape().element_type()));

  // Output tile offsets.
  auto out_offset_batch = b.create<ma::MulIOp>(
      convert_scalar(pid_batch), CreateConst(b, int_ty, stride_out_batch));
  auto out_offsets_m = b.create<ma::MulIOp>(
      b.create<mt::ExpandDimsOp>(convert_range(range_m), 1),
      CreateConst(b, int_ty, stride_out_m, shape_m_1));

  auto out_offsets_n = b.create<ma::MulIOp>(
      b.create<mt::ExpandDimsOp>(convert_range(range_n), 0),
      CreateConst(b, int_ty, stride_out_n, shape_1_n));
  auto out_offsets = b.create<ma::AddIOp>(Splat(b, out_offset_batch, shape_m_1),
                                          out_offsets_m);
  out_offsets = b.create<ma::AddIOp>(
      Broadcast(b, out_offsets.getResult().template cast<TensorValue>(),
                shape_m_n),
      Broadcast(b, out_offsets_n.getResult().template cast<TensorValue>(),
                shape_m_n));

  // Output tile mask: check that the indices are within [M, N].
  auto rm_cmp = b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                     b.create<mt::ExpandDimsOp>(range_m, 1),
                                     CreateConst(b, i32_ty, m, shape_m_1));
  auto rn_cmp = b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                     b.create<mt::ExpandDimsOp>(range_n, 0),
                                     CreateConst(b, i32_ty, n, shape_1_n));
  auto out_mask = b.create<ma::AndIOp>(
      Broadcast(b, rm_cmp.getResult().template cast<TensorValue>(), shape_m_n),
      Broadcast(b, rn_cmp.getResult().template cast<TensorValue>(), shape_m_n));

  // Collect all instructions of the dot's output scope.
  absl::flat_hash_set<const HloInstruction*> to_order;
  {
    std::queue<const HloInstruction*> to_add;
    if (root != dot_instr) {
      to_add.push(root);
    }
    while (!to_add.empty()) {
      const HloInstruction* current = to_add.front();
      for (const HloInstruction* operand : current->operands()) {
        if (!to_order.contains(operand)) {
          if (operand != dot_instr) {
            to_add.push(operand);
          }
        }
      }
      CHECK(to_order.insert(current).second);
      to_add.pop();
    }
  }
  // Order them producers before consumers.
  std::vector<const HloInstruction*> to_emit;
  for (const HloInstruction* hlo :
       dot_instr->parent()->MakeInstructionPostOrder()) {
    if (to_order.contains(hlo)) {
      to_emit.push_back(hlo);
    }
  }
  if (!to_emit.empty()) {
    EmitScope(b, libdevice_path, fn, to_emit, values_out, out_offsets,
              out_mask);
  }

  auto out_offset_split_k = b.create<ma::MulIOp>(
      convert_scalar(pid_k), CreateConst(b, int_ty, stride_out_split_k));
  out_offsets = b.create<ma::AddIOp>(out_offsets,
                                     Splat(b, out_offset_split_k, shape_m_n));
  for (int i = 0;
       i < fn.getNumArguments() - dot_instr->parent()->num_parameters(); ++i) {
    Value out = fn.getArgument(i + dot_instr->parent()->num_parameters());
    const HloInstruction* producer =
        root->shape().IsTuple() ? root->operand(i) : root;
    b.create<mt::StoreOp>(AddPtr(b, Splat(b, out, shape_m_n), out_offsets),
                          values_out[producer], out_mask,
                          mt::CacheModifier::NONE, mt::EvictionPolicy::NORMAL);
  }
  return launch_dimensions;
}
```
# softmax

```c++
StatusOr<LaunchDimensions> SoftMax(
    mlir::OpBuilder builder, absl::string_view libdevice_path,
    const HloComputation* computation, mlir::triton::FuncOp fn,
    const tensorflow::AutotuneResult::TritonGemmKey& config, int) {
  const HloInstruction* root = computation->root_instruction();
  auto loc = mlir::NameLoc::get(builder.getStringAttr(root->name()));
  mlir::ImplicitLocOpBuilder b(loc, builder);

  // Assumptions we make about the matcher:
  //   * matches *exactly* softmax on the last axis, not just something
  //     softmax-like
  //   * the implementation of softmax is like in jax.nn.softmax
  //   * all the shapes have canonical layout (logical layout = physical layout)

  // TODO(bchetioui): generalise to Softmax-like patterns involving elementwise
  // ops.
  // TODO(bchetioui): allow doing several rows per block (e.g. for when rows
  // are smaller than the minimum transaction size)

  CHECK_EQ(root->opcode(), HloOpcode::kDivide);
  CHECK_EQ(root->operand(1)->opcode(), HloOpcode::kBroadcast);

  const HloInstruction* reduce = root->operand(1)->operand(0);
  Shape root_shape = root->shape();

  CHECK_EQ(reduce->opcode(), HloOpcode::kReduce);
  CHECK_EQ(reduce->dimensions().size(), 1);
  CHECK_EQ(reduce->dimensions()[0], root_shape.rank() - 1);

  int row_len = root_shape.dimensions_minor(0);
  int block_row = 1;

  // block_row must be a power of two.
  while (block_row < row_len) {
    block_row *= 2;
  }

  int num_rows = 1;
  for (int minor_axis = 1; minor_axis < root_shape.rank(); ++minor_axis)
    num_rows *= root_shape.dimensions_minor(minor_axis);

  const LaunchDimensions launch_dimensions{
      {num_rows, 1, 1}, {config.num_warps() * WarpSize(), 1, 1}};

  // In the vanilla softmax case, the output type is the same as the input type.
  PrimitiveType root_element_type = root->shape().element_type();
  PrimitiveType producer_element_type =
      computation->parameter_instruction(0)->shape().element_type();

  CHECK_EQ(root_element_type, producer_element_type);

  // We assume that both the input and the result use a floating point data
  // type.
  auto root_ty = TritonType(b, root_element_type).cast<mlir::FloatType>();

  // softmax_kernel(input_ptr, output_ptr, num_rows, row_len, block_row) {
  //   row_index = tl.program_id(0)
  //   row_stride = row_len
  //   offset = row_index * row_stride
  Value row_index = b.create<mt::GetProgramIdOp>(mt::ProgramIDDim::X);
  Value row_stride = b.create<ma::ConstantIntOp>(row_len, /*width=*/32);
  Value offset = b.create<ma::MulIOp>(row_index, row_stride);

  //   input_ptr += offset
  //   output_ptr += offset
  Value input_ptr = AddPtr(b, fn.getArgument(0), offset);
  Value output_ptr = AddPtr(b, fn.getArgument(1), offset);

  //   row_tile = tl.arange(0, block_row)
  Value row_tile = b.create<mt::MakeRangeOp>(
      mlir::RankedTensorType::get(block_row, b.getI32Type()), 0, block_row);

  //   mask = row_tile < row_stride
  Value splat_row_stride = Splat(b, row_stride, block_row);
  Value mask =
      b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, row_tile, splat_row_stride);

  //   row = tl.load(input_ptr + row_tile, mask=row_tile < row_len,
  //                 other=float('-inf'))
  Value splat_input_ptr = Splat(b, input_ptr, block_row);
  Value load_ptrs = AddPtr(b, splat_input_ptr, row_tile);
  llvm::APFloat minus_inf =
      llvm::APFloat::getInf(root_ty.getFloatSemantics(), /*Negative=*/true);

  Value other = Splat(b, b.create<ma::ConstantFloatOp>(minus_inf, root_ty),
                      row_tile.getType().cast<mlir::ShapedType>().getShape());
  Value row =
      b.create<mt::LoadOp>(load_ptrs, mask, other, mt::CacheModifier::NONE,
                           mt::EvictionPolicy::NORMAL, /*isVolatile=*/false);

  //   row_max = tl.max(row, axis=0)
  // Triton actually only performs reductions on float32 inputs, and we must
  // thus upcast/downcast our input if its data type is different.
  Value casted_row = Cast(b, row, b.getF32Type());

  mt::ReduceOp row_max =
      b.create<mt::ReduceOp>(SmallVector<Value>({casted_row}), 0);

  {
    mlir::Block* max_reducer =
        b.createBlock(&row_max->getRegion(0), {},
                      {b.getF32Type(), b.getF32Type()}, {loc, loc});

    b.setInsertionPointToStart(max_reducer);
    // Lowering for MaxFOp from TritonGPU to LLVM is not implemented, so we use
    // select and compare instead.
    Value cmpOp = b.create<ma::CmpFOp>(ma::CmpFPredicate::OGE,
                                       max_reducer->getArgument(0),
                                       max_reducer->getArgument(1));
    Value selectOp = b.create<ma::SelectOp>(cmpOp, max_reducer->getArgument(0),
                                            max_reducer->getArgument(1));

    b.create<mt::ReduceReturnOp>(SmallVector<Value>({selectOp}));
    b.setInsertionPointAfter(row_max);
  }

  //   numerator = tl.exp(row - row_max)
  Value splat_row_max = Splat(b, row_max->getResult(0), block_row);
  Value bounded_row = b.create<ma::SubFOp>(casted_row, splat_row_max);
  Value numerator = b.create<mlir::math::ExpOp>(bounded_row);

  //   denominator = tl.sum(numerator, axis=0)
  mt::ReduceOp denominator =
      b.create<mt::ReduceOp>(SmallVector<Value>({numerator}), 0);

  {
    mlir::Block* sum_reducer =
        b.createBlock(&denominator->getRegion(0), {},
                      {b.getF32Type(), b.getF32Type()}, {loc, loc});

    b.setInsertionPointToStart(sum_reducer);
    Value addOp = b.create<ma::AddFOp>(sum_reducer->getArgument(0),
                                       sum_reducer->getArgument(1));
    b.create<mt::ReduceReturnOp>(SmallVector<Value>({addOp}));
    b.setInsertionPointAfter(denominator);
  }

  //   result = (numerator / denominator).to(output_ptr.dtype.element_ty)
  Value splat_denominator = Splat(b, denominator->getResult(0), block_row);
  Value division = b.create<ma::DivFOp>(numerator, splat_denominator);
  Value result = Cast(b, division, root_ty);

  //   tl.store(output_ptr + row_tile, result, mask=mask)
  Value splat_output_ptr = Splat(b, output_ptr, block_row);
  Value store_ptrs = AddPtr(b, splat_output_ptr, row_tile);

  b.create<mt::StoreOp>(store_ptrs, result, mask, mt::CacheModifier::NONE,
                        mt::EvictionPolicy::NORMAL);
  // }

  return launch_dimensions;
}
```
