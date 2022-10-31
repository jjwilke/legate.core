/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/mapping/operation.h"
#include "core/utilities/deserializer.h"

namespace legate {
namespace mapping {

using LegionTask = Legion::Task;
using LegionCopy = Legion::Copy;
using LegionFill = Legion::Fill;

using namespace Legion;
using namespace Legion::Mapping;

Task::Task(const LegionTask* task,
           const LibraryContext& library,
           MapperRuntime* runtime,
           const MapperContext context)
  : task_(task), library_(library)
{
  TaskDeserializer dez(task, runtime, context);
  inputs_     = dez.unpack<std::vector<Store>>();
  outputs_    = dez.unpack<std::vector<Store>>();
  reductions_ = dez.unpack<std::vector<Store>>();
  scalars_    = dez.unpack<std::vector<Scalar>>();
  dez.unpack<bool>();      // can_raise_exception
  dez.unpack<bool>();      // insert_barrier
  dez.unpack<uint32_t>();  // # communicators
  machine_desc_ = dez.unpack<mapping::MachineDesc>();
  sharding_id_  = dez.unpack<uint32_t>();
}

int64_t Task::task_id() const { return library_.get_local_task_id(task_->task_id); }

TaskTarget Task::target() const
{
  switch (task_->target_proc.kind()) {
    case Processor::LOC_PROC: return TaskTarget::CPU;
    case Processor::TOC_PROC: return TaskTarget::GPU;
    case Processor::OMP_PROC: return TaskTarget::OMP;
    default: {
      assert(false);
    }
  }
  assert(false);
  return TaskTarget::CPU;
}

Copy::Copy(const LegionCopy* copy, MapperRuntime* runtime, const MapperContext context)
  : copy_(copy)
{
  CopyDeserializer dez(copy->mapper_data,
                       copy->mapper_data_size,
                       {copy->src_requirements,
                        copy->dst_requirements,
                        copy->src_indirect_requirements,
                        copy->dst_indirect_requirements},
                       runtime,
                       context);
  inputs_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  outputs_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  input_indirections_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  output_indirections_ = dez.unpack<std::vector<Store>>();
  machine_desc_        = dez.unpack<mapping::MachineDesc>();
#ifdef DEBUG_LEGATE
  for (auto& input : inputs_) assert(!input.is_future());
  for (auto& output : outputs_) assert(!output.is_future());
  for (auto& input_indirection : input_indirections_) assert(!input_indirection.is_future());
  for (auto& output_indirection : output_indirections_) assert(!output_indirection.is_future());
#endif
}

Fill::Fill(const LegionFill* fill) : fill_(fill)
{
  FillDeserializer dez(fill->mapper_data, fill->mapper_data_size);
  machine_desc_ = dez.unpack<mapping::MachineDesc>();
  sharding_id_  = dez.unpack<uint32_t>();
}

}  // namespace mapping
}  // namespace legate
