#include "core/mapping/base_mapper.h"
#include <legion/legion_config.h>
#include <legion/legion_ops.h>
#include <legion/legion_types.h>
#include <legion/legion_utilities.h>
#include <legion/mapper_manager.h>
#include <legion/runtime.h>
#include <realm/machine.h>
#include <algorithm>
#include <cstdint>
#include "core/mapping/mapping.h"
#include "core/runtime/context.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/typedefs.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "legion.h"
#include "legion/legion_context.h"
#include "legion/legion_tasks.h"

namespace legate {
namespace mapping {

static constexpr int kFID_A   = 0;
static constexpr int kFID_B   = 1;
static constexpr int kFID_C   = 2;
static constexpr int kTASK_ID = 0;

Legion::Processor GetTestProc();

class TestMapper : public mapping::BaseMapper {
 public:
  TestMapper(Legion::Runtime* rt,
             Legion::Machine m,
             const LibraryContext& ctx,
             const BaseMapperConfig& config)
    : BaseMapper(rt, m, ctx, config)
  {
  }

  bool is_pure() const override { return true; }

  TaskTarget task_target(const Task& task, const std::vector<TaskTarget>& options) override
  {
    return options[0];
  }

  Scalar tunable_value(TunableID tunable_id) override { return 0; }
};

class TestMapperManager {
 public:
  TestMapperManager(TestMapper* mapper)
    : mapper_(mapper),
      manager_(Legion::Runtime::get_runtime()->runtime,
               mapper,
               /*mapper_id=*/0,
               GetTestProc(),
               /*reentrant=*/true)
  {
  }

  TestMapper* mapper() { return mapper_; }

  auto* manager() { return &manager_; }

 private:
  TestMapper* mapper_;
  Legion::Internal::SerializingManager manager_;
};

struct Wrapper {
  bool value;
};

struct TestStoreConfig {
  bool is_output              = false;
  int32_t dim                 = 1;
  legate::LegateTypeCode code = legate::LegateTypeCode::DOUBLE_LT;
  int32_t fid;
};

struct TestTask {
  std::unique_ptr<Legion::Internal::IndividualTask> task;
  std::unique_ptr<Legion::Internal::InnerContext> inner_ctx;
  std::unique_ptr<Legion::TaskLauncher> launcher;
  Legion::Future future;
};

struct TestStore {
  Legion::LogicalRegion region;
  int32_t fid;
};

Legion::Internal::InnerContext* GetLegionTopContext()
{
  static Legion::Internal::InnerContext* top_ctx = nullptr;
  if (!top_ctx) {
    int argc     = 1;
    char* argv[] = {const_cast<char*>("test")};
    Legion::Runtime::start(argc, argv, /*background=*/true);

    auto* runtime          = Legion::Runtime::get_runtime();
    auto top_level_task_id = runtime->generate_library_task_ids("legion_canonical_python", 3);
    auto* ctx              = runtime->begin_implicit_task(top_level_task_id,
                                             0 /*mapper id*/,
                                             Legion::Processor::LOC_PROC,
                                             "legion_top_level_task",
                                             /*control_replicate=*/true);
    top_ctx                = dynamic_cast<Legion::Internal::InnerContext*>(ctx);
  }
  return top_ctx;
}

Legion::Processor GetTestProc()
{
  auto* runtime = Legion::Runtime::get_runtime();
  return runtime->runtime->GetProxyProcessor(Realm::Processor::LOC_PROC);
}

TestMapperManager GetTestMapperManager(const BaseMapperConfig& mapper_config)
{
  auto machine             = Realm::Machine::get_machine();
  Legion::Runtime* runtime = Legion::Runtime::get_runtime();
  ResourceConfig resource_config{
    .max_tasks         = 64,
    .max_mappers       = 1,
    .max_reduction_ops = 0,
  };
  LibraryContext context(runtime, "test", resource_config);
  auto* mapper = new TestMapper(runtime, machine, context, mapper_config);
  return TestMapperManager(mapper);
}

static void dummy_task_wrapper(
  const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p)
{
}

void AppendStore(Legion::Serializer& ser, const TestStoreConfig& config)
{
  ser.serialize(Wrapper{.value = false});             // is future
  ser.serialize(Wrapper{.value = config.is_output});  // is output
  ser.serialize(config.dim);                          // dim
  ser.serialize(legate::LegateTypeCode::DOUBLE_LT);   // code
  ser.serialize(int32_t(-1));                         // no transform
  ser.serialize(int32_t(-1));                         // redop
  ser.serialize(config.dim);                          // dim
  ser.serialize(uint32_t(0));                         // zeroth region
  ser.serialize(config.fid);                          // FID
}

void MakeTaskImpl(int task_id)
{
  static std::map<int, Legion::Internal::TaskImpl*> impls;

  auto* task_impl = impls[task_id];
  if (task_impl == nullptr) {
    Legion::TaskVariantRegistrar registrar(task_id);
    registrar.global_registration = false;

    auto* runtime         = Legion::Runtime::get_runtime();
    task_impl             = runtime->runtime->find_or_create_task_impl(task_id);
    Legion::VariantID vid = 1;
    Legion::CodeDescriptor descriptor(dummy_task_wrapper);
    Legion::Internal::VariantImpl variant(runtime->runtime,
                                          vid,
                                          task_impl,
                                          registrar,
                                          /*return_type_size=*/0,
                                          /*has_return_type_size=*/false,
                                          descriptor);
    task_impl->add_variant(&variant);
    impls[task_id] = task_impl;
  }
}

Legion::LogicalRegion MakeTestLogicalRegion()
{
  // Legion::TaskID tid = 1;
  Legion::Rect<1> elem_rect(0, 10);
  auto* runtime = Legion::Runtime::get_runtime();
  auto* top_ctx = GetLegionTopContext();
  auto is       = runtime->create_index_space(top_ctx, elem_rect);
  auto fs       = runtime->create_field_space(top_ctx);

  Legion::FieldAllocator allocator = runtime->create_field_allocator(top_ctx, fs);
  allocator.allocate_field(sizeof(double), kFID_A);
  runtime->attach_name(fs, kFID_A, "FIELD A");
  allocator.allocate_field(sizeof(double), kFID_B);
  runtime->attach_name(fs, kFID_B, "FIELD B");
  allocator.allocate_field(sizeof(double), kFID_C);
  runtime->attach_name(fs, kFID_C, "FIELD C");
  return runtime->create_logical_region(top_ctx, is, fs);
}

TestTask MakeTestTask(int task_id, std::vector<TestStore> stores)
{
  MakeTaskImpl(task_id);

  std::vector<Legion::RegionRequirement> reqs;
  std::vector<Legion::OutputRequirement> output_reqs;
  std::vector<unsigned> parent_indexes;
  std::vector<bool> virt_mapped;

  auto* inner_ctx = GetLegionTopContext();
  auto* runtime   = Legion::Runtime::get_runtime();

  TestTask test_task;

  test_task.task = std::make_unique<Legion::Internal::IndividualTask>(runtime->runtime);
  test_task.task->activate();
  test_task.task->initialize_operation(inner_ctx, /*track=*/false);

  test_task.inner_ctx =
    std::make_unique<Legion::Internal::InnerContext>(runtime->runtime,
                                                     test_task.task.get(),
                                                     /*depth=*/0,
                                                     /*full_inner=*/false,
                                                     reqs,
                                                     output_reqs,
                                                     parent_indexes,
                                                     virt_mapped,
                                                     Legion::Internal::ApEvent::NO_AP_EVENT);

  Legion::Serializer task_ser;
  task_ser.serialize(uint32_t(stores.size()));  // inputs
  for (auto& store : stores) { AppendStore(task_ser, {.fid = store.fid}); }

  task_ser.serialize(uint32_t(0));  // outputs
  task_ser.serialize(uint32_t(0));  // reductions
  task_ser.serialize(uint32_t(0));  // scalars

  Legion::UntypedBuffer task_buffer(task_ser.get_buffer(), task_ser.get_buffer_size());

  Legion::Serializer mapper_arg_ser;
  mapper_arg_ser.serialize(uint32_t(legate::mapping::TaskTarget::CPU));  // preferred kind
  mapper_arg_ser.serialize(uint32_t(1));                                 // num ranges
  mapper_arg_ser.serialize(uint32_t(legate::mapping::TaskTarget::CPU));  // kind
  mapper_arg_ser.serialize(uint32_t(1));                                 // per-node count
  mapper_arg_ser.serialize(uint32_t(0));                                 // lo
  mapper_arg_ser.serialize(uint32_t(0));                                 // hi

  Legion::UntypedBuffer map_buffer(mapper_arg_ser.get_buffer(), mapper_arg_ser.get_buffer_size());
  test_task.launcher = std::make_unique<Legion::TaskLauncher>(
    task_id, task_buffer, Legion::Predicate::TRUE_PRED, /*id=*/0, /*tag=*/0, map_buffer);

  std::map<Legion::RegionTreeID, Legion::RegionRequirement> store_requirements;
  for (auto& store : stores) {
    auto [iter, inserted] = store_requirements.try_emplace(
      store.region.get_tree_id(), store.region, READ_WRITE, EXCLUSIVE, store.region);
    iter->second.add_field(store.fid);
  }

  for (auto& [id, req] : store_requirements) { test_task.launcher->add_region_requirement(req); }

  test_task.future = test_task.task->initialize_task(
    test_task.inner_ctx.get(), *test_task.launcher, new Legion::Internal::Provenance("test"));
  test_task.task->set_target_proc(GetTestProc());

  return test_task;
};

class BaseMapperTest : public ::testing::Test {
 public:
  void SetUp() override
  {
    GetLegionTopContext();  // ensure runtime is inited
  }
};

class MapTaskContext {
 public:
  MapTaskContext(TestMapperManager* manager, std::vector<TestStore> stores)
    : task_(MakeTestTask(kTASK_ID, stores)), manager_(manager)
  {
    info_ = manager_->manager()->begin_mapper_call(
      Legion::Internal::MAP_TASK_CALL, task_.task.get(), continuation_precondition_);
    manager->mapper()->map_task(info_, *task_.task, input, output);
  }

  ~MapTaskContext() { manager_->manager()->finish_mapper_call(info_); }

  Legion::Mapping::Mapper::MapTaskOutput output;
  Legion::Mapping::Mapper::MapTaskInput input;

 private:
  TestTask task_;
  Legion::Internal::MappingCallInfo* info_;
  Legion::Internal::RtEvent continuation_precondition_;
  TestMapperManager* manager_;
};

TEST_F(BaseMapperTest, BasicReuseTest)
{
  Legion::LogicalRegion lr = MakeTestLogicalRegion();

  auto manager =
    GetTestMapperManager({.default_contiguous = false, .single_store_per_mapping = false});

  unsigned long instance_id = 0;
  {
    MapTaskContext ctx(&manager,
                       {{.region = lr, .fid = kFID_A},
                        {.region = lr, .fid = kFID_B},
                        {.region = lr, .fid = kFID_C}});
    ASSERT_EQ(ctx.output.chosen_instances.size(), 1);
    ASSERT_EQ(ctx.output.chosen_instances[0].size(), 1);
    instance_id = ctx.output.chosen_instances[0][0].get_instance_id();
  }

  {
    MapTaskContext ctx(&manager, {{.region = lr, .fid = kFID_A}, {.region = lr, .fid = kFID_B}});
    // These should all reuse the instance created in the first task
    for (auto& instance_vector : ctx.output.chosen_instances) {
      for (auto& instance : instance_vector) { EXPECT_EQ(instance.get_instance_id(), instance_id); }
    }
  }

  {
    MapTaskContext ctx(&manager, {{.region = lr, .fid = kFID_A}, {.region = lr, .fid = kFID_C}});
    // These should all reuse the instance created in the first task
    for (auto& instance_vector : ctx.output.chosen_instances) {
      for (auto& instance : instance_vector) { EXPECT_EQ(instance.get_instance_id(), instance_id); }
    }
  }
}

TEST_F(BaseMapperTest, ContiguousFieldsMatch)
{
  Legion::LogicalRegion lr = MakeTestLogicalRegion();

  auto manager =
    GetTestMapperManager({.default_contiguous = true, .single_store_per_mapping = false});

  unsigned long instance_id = 0;
  {
    MapTaskContext ctx(&manager,
                       {{.region = lr, .fid = kFID_A},
                        {.region = lr, .fid = kFID_B},
                        {.region = lr, .fid = kFID_C}});
    ASSERT_EQ(ctx.output.chosen_instances.size(), 1);
    ASSERT_EQ(ctx.output.chosen_instances[0].size(), 1);
    instance_id = ctx.output.chosen_instances[0][0].get_instance_id();
  }

  {
    MapTaskContext ctx(&manager, {{.region = lr, .fid = kFID_A}, {.region = lr, .fid = kFID_B}});
    // These should all reuse the instance created in the first task
    for (auto& instance_vector : ctx.output.chosen_instances) {
      for (auto& instance : instance_vector) { EXPECT_EQ(instance.get_instance_id(), instance_id); }
    }
  }

  {
    MapTaskContext ctx(&manager, {{.region = lr, .fid = kFID_A}, {.region = lr, .fid = kFID_C}});
    // These should not reuse the instance created in the first task
    // since A and C will not be a contiguous subset
    for (auto& instance_vector : ctx.output.chosen_instances) {
      for (auto& instance : instance_vector) { EXPECT_NE(instance.get_instance_id(), instance_id); }
    }
  }
}

TEST_F(BaseMapperTest, SingleStorePerMappingReuse)
{
  Legion::LogicalRegion lr = MakeTestLogicalRegion();

  using ::testing::ElementsAreArray;

  auto manager =
    GetTestMapperManager({.default_contiguous = true, .single_store_per_mapping = true});

  std::vector<unsigned long> instance_ids;
  {
    MapTaskContext ctx(&manager,
                       {{.region = lr, .fid = kFID_A},
                        {.region = lr, .fid = kFID_B},
                        {.region = lr, .fid = kFID_C}});
    // These should all be on a single RegionRequirement
    ASSERT_EQ(ctx.output.chosen_instances.size(), 1);
    for (auto& instance_vec : ctx.output.chosen_instances) {
      // Each store should be backed by its own instance
      ASSERT_EQ(instance_vec.size(), 3);
      for (auto& instance : instance_vec) { instance_ids.push_back(instance.get_instance_id()); }
    }
  }

  {
    MapTaskContext ctx(&manager,
                       {{.region = lr, .fid = kFID_A},
                        {.region = lr, .fid = kFID_B},
                        {.region = lr, .fid = kFID_C}});
    ASSERT_EQ(ctx.output.chosen_instances.size(), 1);
    // These should all reuse the instances created in the first task
    std::vector<unsigned long> test_ids;
    for (auto& instance_vec : ctx.output.chosen_instances) {
      // Each store should be back by a single instance
      ASSERT_EQ(instance_vec.size(), 3);
      for (auto& instance : instance_vec) { test_ids.push_back(instance.get_instance_id()); }
    }
    EXPECT_THAT(test_ids, ElementsAreArray(instance_ids));
  }
}

TEST_F(BaseMapperTest, SingleStorePerMappingNonContiguousReuse)
{
  Legion::LogicalRegion lr = MakeTestLogicalRegion();

  using ::testing::ElementsAreArray;

  auto manager =
    GetTestMapperManager({.default_contiguous = true, .single_store_per_mapping = true});

  std::vector<unsigned long> instance_ids;
  {
    MapTaskContext ctx(&manager,
                       {{.region = lr, .fid = kFID_A},
                        {.region = lr, .fid = kFID_B},
                        {.region = lr, .fid = kFID_C}});
    // These should all be on a single RegionRequirement
    ASSERT_EQ(ctx.output.chosen_instances.size(), 1);
    for (auto& instance_vec : ctx.output.chosen_instances) {
      // Each store should be backed by its own instance
      ASSERT_EQ(instance_vec.size(), 3);
      for (auto& instance : instance_vec) { instance_ids.push_back(instance.get_instance_id()); }
    }
  }

  {
    MapTaskContext ctx(&manager,
                       {{.region = lr, .fid = kFID_A},
                        {.region = lr, .fid = kFID_B},
                        {.region = lr, .fid = kFID_C}});
    ASSERT_EQ(ctx.output.chosen_instances.size(), 1);
    // These should all reuse the instances created in the first task
    std::vector<unsigned long> test_ids;
    for (auto& instance_vec : ctx.output.chosen_instances) {
      // Each store should be back by a single instance
      ASSERT_EQ(instance_vec.size(), 3);
      for (auto& instance : instance_vec) { test_ids.push_back(instance.get_instance_id()); }
    }
    EXPECT_THAT(test_ids, ElementsAreArray(instance_ids));
  }
}

}  // namespace mapping
}  // namespace legate

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int rc        = RUN_ALL_TESTS();
  auto* runtime = Legion::Runtime::get_runtime();
  runtime->finish_implicit_task(legate::mapping::GetLegionTopContext());
  Legion::Runtime::wait_for_shutdown();
  return rc;
}