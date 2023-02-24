#include <legion/legion_config.h>
#include <legion/legion_types.h>
#include <legion/legion_utilities.h>
#include <legion/runtime.h>
#include <realm/machine.h>
#include <algorithm>
#include <cstdint>
#include "core/mapping/mapping.h"
#include "core/runtime/context.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/typedefs.h"
#include "gtest/gtest.h"
#include "core/mapping/base_mapper.h"
#include "legion.h"
#include "legion/legion_tasks.h"
#include "legion/legion_context.h"

namespace legate {
namespace mapping {

class TestMapper : public mapping::BaseMapper {
 public:
  TestMapper(Legion::Runtime* rt, Legion::Machine m, const LibraryContext& ctx) : group_fields_(false), BaseMapper(rt, m, ctx){
  }

  bool is_pure() const override {
    return true;
  }

  void set_group_fields(bool flag) {
    group_fields_ = flag;
  }

  TaskTarget task_target(const Task& task, const std::vector<TaskTarget>& options) override {
    return options[0];
  }

  std::vector<StoreMapping> store_mappings(const Task& task,
                                           const std::vector<StoreTarget>& options) override {

    static constexpr uint32_t unassigned{-1U};
    size_t max_num_requirements =
      task.inputs().size() + task.outputs().size() + task.reductions().size();
    std::vector<int64_t> requirement_to_mapping(max_num_requirements, -1);
    std::vector<StoreMapping> mappings;

    if (!group_fields_){
      return mappings;
    }

    auto visit = [&](const legate::mapping::Store& store) {
      auto idx = store.requirement_index();
      if (idx == unassigned) {
        mappings.push_back(StoreMapping::default_mapping(store, options.front()));
      } else if (requirement_to_mapping[idx] == -1) {
        std::cout << "New index " << idx << "->" << mappings.size() << std::endl;
        requirement_to_mapping[idx] = mappings.size();
        mappings.push_back(StoreMapping::default_mapping(store, options.front()));
      } else {
        std::cout << "Reuse index " << idx << "->" << requirement_to_mapping[idx] << std::endl;
        auto& mapping = mappings[requirement_to_mapping[idx]];
        mapping.stores.push_back(store);
      }
    };

    for (auto& input : task.inputs()) { visit(input); }
    for (auto& output : task.outputs()) { visit(output); }

    return mappings;
  }

  Scalar tunable_value(TunableID tunable_id) override {
    return 0;
  }

 private:
  bool group_fields_;
};

struct Wrapper {
  bool value;
};

Legion::Internal::InnerContext* GetLegionTopContext(){
  static Legion::Internal::InnerContext* top_ctx = nullptr;
  if (!top_ctx){
    int argc = 1;
    char* argv[] = { const_cast<char*>("test") };
    Legion::Runtime::start(argc, argv, /*background=*/true);

    auto *runtime = Legion::Runtime::get_runtime();
    auto top_level_task_id = runtime->generate_library_task_ids("legion_canonical_python", 3);
    auto* ctx = runtime->begin_implicit_task(top_level_task_id,
                                      0 /*mapper id*/,
                                      Legion::Processor::LOC_PROC,
                                      "legion_top_level_task",
                                      /*control_replicate=*/true);
    top_ctx = dynamic_cast<Legion::Internal::InnerContext*>(ctx);
  }
  return top_ctx;
}

TestMapper* GetTestMapper(){
  auto machine = Realm::Machine::get_machine();
  Legion::Runtime* runtime = Legion::Runtime::get_runtime();
  ResourceConfig config{
    .max_tasks         = 64,
    .max_mappers       = 1,
    .max_reduction_ops = 0,
  };
  LibraryContext context(runtime, "test", config);
  return new TestMapper(runtime, machine, context);
}

class BaseMapperTest : public ::testing::Test {
 public:
  void SetUp() override {
    GetLegionTopContext(); //ensure runtime is inited
  }

};

static void dummy_task_wrapper(const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p){

}

Legion::Processor GetTestProc(){
  auto* runtime = Legion::Runtime::get_runtime();
  return runtime->runtime->GetProxyProcessor(Realm::Processor::LOC_PROC);
}

struct TestTask {
  std::unique_ptr<Legion::Internal::IndividualTask> task;
  std::unique_ptr<Legion::Internal::InnerContext> inner_ctx;
  std::unique_ptr<Legion::TaskLauncher> launcher;
  Legion::Future future;
};

static constexpr int FID_A = 0;
static constexpr int FID_B = 1;
static constexpr int FID_C = 2;

static constexpr int TASK_ID = 0;

struct TestStore {
  Legion::LogicalRegion region;
  int32_t fid;
};

struct StoreConfig {
  bool is_output = false;
  int32_t dim = 1;
  legate::LegateTypeCode code = legate::LegateTypeCode::DOUBLE_LT;
  int32_t fid;
};

void AppendStore(Legion::Serializer& ser, const StoreConfig& config){
  ser.serialize(Wrapper{ .value = false }); //is future
  ser.serialize(Wrapper{ .value = config.is_output }); //is output
  ser.serialize(config.dim); // dim
  ser.serialize(legate::LegateTypeCode::DOUBLE_LT); // code
  ser.serialize(int32_t(-1)); // no transform
  ser.serialize(int32_t(-1)); //redop
  ser.serialize(config.dim); //dim
  ser.serialize(uint32_t(0)); // zeroth region
  ser.serialize(config.fid); // FID
}

void MakeTaskImpl(int task_id) {
  static std::map<int, Legion::Internal::TaskImpl*> impls;

  auto* task_impl = impls[task_id];
  if (task_impl == nullptr){
    Legion::TaskVariantRegistrar registrar(task_id);
    registrar.global_registration = false;

    auto* runtime = Legion::Runtime::get_runtime();
    task_impl = runtime->runtime->find_or_create_task_impl(task_id);
    Legion::VariantID vid = 1;
    Legion::CodeDescriptor descriptor(dummy_task_wrapper);
    Legion::Internal::VariantImpl variant(runtime->runtime, vid, task_impl, registrar, /*return_type_size=*/0, /*has_return_type_size=*/false, descriptor);
    task_impl->add_variant(&variant);
    impls[task_id] = task_impl;
  }
}

TestTask MakeTestTask(int task_id, std::vector<TestStore> stores){
  MakeTaskImpl(task_id);

  std::vector<Legion::RegionRequirement> reqs;
  std::vector<Legion::OutputRequirement> output_reqs;
  std::vector<unsigned> parent_indexes;
  std::vector<bool> virt_mapped;

  auto* inner_ctx = GetLegionTopContext();
  auto* runtime = Legion::Runtime::get_runtime();

  TestTask test_task;

  test_task.task = std::make_unique<Legion::Internal::IndividualTask>(runtime->runtime);
  test_task.task->activate();
  test_task.task->initialize_operation(inner_ctx, /*track=*/false);

  test_task.inner_ctx = std::make_unique<Legion::Internal::InnerContext>(runtime->runtime, test_task.task.get(), /*depth=*/0, /*full_inner=*/false,
    reqs, output_reqs, parent_indexes, virt_mapped, Legion::Internal::ApEvent::NO_AP_EVENT);


  Legion::Serializer task_ser;
  task_ser.serialize(uint32_t(stores.size())); //inputs
  for (auto& store : stores){
    AppendStore(task_ser, { .fid = store.fid} );
  }

  task_ser.serialize(uint32_t(0)); //outputs
  task_ser.serialize(uint32_t(0)); //reductions
  task_ser.serialize(uint32_t(0)); //scalars

  Legion::UntypedBuffer task_buffer(task_ser.get_buffer(), task_ser.get_buffer_size());


  Legion::Serializer mapper_arg_ser;
  mapper_arg_ser.serialize(uint32_t(legate::mapping::TaskTarget::CPU)); // preferred kind
  mapper_arg_ser.serialize(uint32_t(1)); // num ranges
  mapper_arg_ser.serialize(uint32_t(legate::mapping::TaskTarget::CPU)); // kind
  mapper_arg_ser.serialize(uint32_t(1)); //per-node count
  mapper_arg_ser.serialize(uint32_t(0)); // lo
  mapper_arg_ser.serialize(uint32_t(0)); // hi

  Legion::UntypedBuffer map_buffer(mapper_arg_ser.get_buffer(), mapper_arg_ser.get_buffer_size());
  test_task.launcher = std::make_unique<Legion::TaskLauncher>(task_id, task_buffer, Legion::Predicate::TRUE_PRED, /*id=*/0, /*tag=*/0, map_buffer);

  std::map<Legion::RegionTreeID, Legion::RegionRequirement> store_requirements;
  for (auto& store : stores){
    auto [iter, inserted] = store_requirements.try_emplace(store.region.get_tree_id(), store.region, READ_WRITE, EXCLUSIVE, store.region);
    iter->second.add_field(store.fid);
  }

  for (auto& [id, req] : store_requirements){
    std::cout << "Adding region requirement with " << req.instance_fields.size() << " fields" << std::endl;
    test_task.launcher->add_region_requirement(req);
  }

  test_task.future = test_task.task->initialize_task(test_task.inner_ctx.get(), *test_task.launcher, new Legion::Internal::Provenance("test"));
  test_task.task->set_target_proc(GetTestProc());

  return test_task;
};

TEST_F(BaseMapperTest, DoNothingTest){
  auto* runtime = Legion::Runtime::get_runtime();
  auto* top_ctx = GetLegionTopContext();
  Legion::TaskID tid = 1;
  Legion::Rect<1> elem_rect(0,10);
  auto is = runtime->create_index_space(top_ctx, elem_rect);
  auto fs = runtime->create_field_space(top_ctx);

  Legion::FieldAllocator allocator = runtime->create_field_allocator(top_ctx, fs);
  allocator.allocate_field(sizeof(double), FID_A);
  runtime->attach_name(fs, FID_A, "FIELD A");
  allocator.allocate_field(sizeof(double), FID_B);
  runtime->attach_name(fs, FID_B, "FIELD B");
  allocator.allocate_field(sizeof(double), FID_C);
  runtime->attach_name(fs, FID_C, "FIELD C");
  Legion::LogicalRegion lr = runtime->create_logical_region(top_ctx, is, fs);

  Legion::RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);

  //auto proc = runtime->runtime->GetProxyProcessor(Realm::Processor::LOC_PROC);

  auto* mapper = GetTestMapper();
  Legion::Internal::SerializingManager mapper_manager(runtime->runtime, mapper,
    /*mapper_id=*/0, GetTestProc(), /*reentrant=*/true);

  auto* inner_ctx = GetLegionTopContext();

  auto task = MakeTestTask(TASK_ID, {
    {.region = lr, .fid = FID_A},
    {.region = lr, .fid = FID_B},
    {.region = lr, .fid = FID_C}
  });

  Legion::Internal::RtEvent continuation_precondition;
  auto* info = mapper_manager.begin_mapper_call(Legion::Internal::MAP_TASK_CALL, task.task.get(), continuation_precondition);

  {
    std::cout << "\nTrying to find reused grouped fields" << std::endl;
    Legion::Mapping::Mapper::MapTaskOutput output;
    Legion::Mapping::Mapper::MapTaskInput input;
    mapper->set_group_fields(true);
    mapper->map_task(info, *task.task, input, output);

    for (auto& instance_vector : output.chosen_instances){
      for (auto& instance : instance_vector){
        std::cout << "Chose instance " << instance.get_instance_id() << std::endl;
      }
    }
    mapper->set_group_fields(false);
  }

  {
    std::cout << "\nInitial creationg with individual fields" << std::endl;
    Legion::Mapping::Mapper::MapTaskOutput output;
    Legion::Mapping::Mapper::MapTaskInput input;
    mapper->map_task(info, *task.task, input, output);

    for (auto& instance_vector : output.chosen_instances){
      for (auto& instance : instance_vector){
        std::cout << "Chose instance " << instance.get_instance_id() << std::endl;
      }
    }
  }

  {
    std::cout << "\nTrying to find reused individual fields" << std::endl;
    Legion::Mapping::Mapper::MapTaskOutput output;
    Legion::Mapping::Mapper::MapTaskInput input;
    mapper->map_task(info, *task.task, input, output);

    for (auto& instance_vector : output.chosen_instances){
      for (auto& instance : instance_vector){
        std::cout << "Chose instance " << instance.get_instance_id() << std::endl;
      }
    }
  }

  {
    mapper->set_group_fields(true);
    std::cout << "\nTrying to find reused grouped fields again" << std::endl;
    Legion::Mapping::Mapper::MapTaskOutput output;
    Legion::Mapping::Mapper::MapTaskInput input;
    mapper->set_group_fields(true);
    mapper->map_task(info, *task.task, input, output);

    for (auto& instance_vector : output.chosen_instances){
      for (auto& instance : instance_vector){
        std::cout << "Chose instance " << instance.get_instance_id() << std::endl;
      }
    }
    mapper->set_group_fields(false);
  }

  mapper_manager.finish_mapper_call(info);

  auto subset_task = MakeTestTask(TASK_ID, {
    {.region = lr, .fid = FID_A},
    {.region = lr, .fid = FID_C}
  });
  info = mapper_manager.begin_mapper_call(Legion::Internal::MAP_TASK_CALL, subset_task.task.get(), continuation_precondition);

  {
    mapper->set_group_fields(true);
    std::cout << "\nTrying to find reused grouped fields in subset task" << std::endl;
    Legion::Mapping::Mapper::MapTaskOutput output;
    Legion::Mapping::Mapper::MapTaskInput input;
    mapper->set_group_fields(true);
    mapper->map_task(info, *subset_task.task, input, output);

    for (auto& instance_vector : output.chosen_instances){
      for (auto& instance : instance_vector){
        std::cout << "Chose instance " << instance.get_instance_id() << std::endl;
      }
    }
    mapper->set_group_fields(false);
  }
}

}
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  auto *runtime = Legion::Runtime::get_runtime();
  runtime->finish_implicit_task(legate::mapping::GetLegionTopContext());
  Legion::Runtime::wait_for_shutdown();
  return rc;
}