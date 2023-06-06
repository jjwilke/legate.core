/* Copyright 2021 NVIDIA Corporation
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

#include <numeric>

#include "core/data/detail/logical_store.h"
#include "core/data/logical_store.h"
#include "core/data/store.h"
#include "core/partitioning/partition.h"
#include "core/type/type_traits.h"
#include "core/utilities/buffer_builder.h"
#include "core/utilities/dispatch.h"
#include "legate_defines.h"

namespace legate {

extern Logger log_legate;

LogicalStore::LogicalStore(std::shared_ptr<detail::LogicalStore>&& impl) : impl_(std::move(impl)) {}

int32_t LogicalStore::dim() const { return impl_->dim(); }

const Type& LogicalStore::type() const { return impl_->type(); }

const Shape& LogicalStore::extents() const { return impl_->extents(); }

size_t LogicalStore::volume() const { return impl_->volume(); }

bool LogicalStore::unbound() const { return impl_->unbound(); }

bool LogicalStore::transformed() const { return impl_->transformed(); }

LogicalStore LogicalStore::promote(int32_t extra_dim, size_t dim_size) const
{
  return LogicalStore(impl_->promote(extra_dim, dim_size));
}

LogicalStore LogicalStore::project(int32_t dim, int64_t index) const
{
  return LogicalStore(impl_->project(dim, index));
}

LogicalStorePartition LogicalStore::partition_by_tiling(std::vector<size_t> tile_shape) const
{
  return LogicalStorePartition(impl_->partition_by_tiling(Shape(std::move(tile_shape))));
}

LogicalStore LogicalStore::slice(int32_t dim, Slice sl) const
{
  return LogicalStore(impl_->slice(dim, sl));
}

LogicalStore LogicalStore::transpose(std::vector<int32_t>&& axes) const
{
  return LogicalStore(impl_->transpose(std::move(axes)));
}

LogicalStore LogicalStore::delinearize(int32_t dim, std::vector<int64_t>&& sizes) const
{
  return LogicalStore(impl_->delinearize(dim, std::move(sizes)));
}

std::shared_ptr<Store> LogicalStore::get_physical_store(LibraryContext* context)
{
  return impl_->get_physical_store(context);
}

void LogicalStore::set_key_partition(const mapping::MachineDesc& machine,
                                     const Partition* partition)
{
  impl_->set_key_partition(machine, partition);
}

LogicalStorePartition::LogicalStorePartition(std::shared_ptr<detail::LogicalStorePartition>&& impl)
  : impl_(std::move(impl))
{
}

LogicalStore LogicalStorePartition::store() const { return LogicalStore(impl_->store()); }

std::shared_ptr<Partition> LogicalStorePartition::partition() const
{
  return impl_->storage_partition()->partition();
}

}  // namespace legate
