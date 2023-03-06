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

#pragma once

#include "core/data/scalar.h"
#include "core/mapping/store.h"

namespace legate {
namespace mapping {

class Task;

// NOTE: codes are chosen to reflect the precendece between the processor kinds
enum class TaskTarget : int32_t {
  GPU = 1,
  OMP = 2,
  CPU = 3,
};

enum class StoreTarget : int32_t {
  SYSMEM    = 1,
  FBMEM     = 2,
  ZCMEM     = 3,
  SOCKETMEM = 4,
};

enum class AllocPolicy : int32_t {
  MAY_ALLOC  = 1,
  MUST_ALLOC = 2,
};

enum class InstLayout : int32_t {
  SOA = 1,
  AOS = 2,
};

struct DimOrdering {
 public:
  enum class Kind : int32_t {
    C       = 1,
    FORTRAN = 2,
    CUSTOM  = 3,
  };

 public:
  DimOrdering() {}

 public:
  DimOrdering(const DimOrdering&)            = default;
  DimOrdering& operator=(const DimOrdering&) = default;

 public:
  DimOrdering(DimOrdering&&)            = default;
  DimOrdering& operator=(DimOrdering&&) = default;

 public:
  bool operator==(const DimOrdering&) const;

 public:
  void populate_dimension_ordering(const Store& store,
                                   std::vector<Legion::DimensionKind>& ordering) const;

 public:
  void c_order();
  void fortran_order();
  void custom_order(std::vector<int32_t>&& dims);

 public:
  Kind kind{Kind::C};
  // When relative is true, 'dims' specifies the order of dimensions
  // for the store's local coordinate space, which will be mapped
  // back to the root store's original coordinate space.
  bool relative{false};
  // Used only when the kind is CUSTOM
  std::vector<int32_t> dims{};
};

struct InstanceMappingPolicy {
 public:
  StoreTarget target{StoreTarget::SYSMEM};
  AllocPolicy allocation{AllocPolicy::MAY_ALLOC};
  InstLayout layout{InstLayout::SOA};
  DimOrdering ordering{};
  bool exact{false};
  bool contiguous{true};
  bool inorder{true};

 public:
  InstanceMappingPolicy() {}

 public:
  InstanceMappingPolicy(const InstanceMappingPolicy&)            = default;
  InstanceMappingPolicy& operator=(const InstanceMappingPolicy&) = default;

 public:
  InstanceMappingPolicy(InstanceMappingPolicy&&)            = default;
  InstanceMappingPolicy& operator=(InstanceMappingPolicy&&) = default;

 public:
  bool operator==(const InstanceMappingPolicy&) const;
  bool operator!=(const InstanceMappingPolicy&) const;

 public:
  void populate_layout_constraints(const Store& store,
                                   Legion::LayoutConstraintSet& layout_constraints) const;

 public:
  static InstanceMappingPolicy default_policy(StoreTarget target, bool exact = false);
};

template <class T>
size_t store_hash_combine(size_t hash, T t)
{
  return std::hash<T>()(t) ^ hash;
}

struct StoreMapping {
 public:
  std::vector<Store> stores{};
  InstanceMappingPolicy policy;

  static constexpr uint32_t kNoGroup = 0;

 public:
  struct StoreMappingConfig {
    StoreTarget target;
    bool exact        = false;
    uint32_t group_id = kNoGroup;
  };
  StoreMapping(uint32_t group_id = kNoGroup) : group_id_(group_id) {}

 public:
  StoreMapping(const StoreMapping&)            = default;
  StoreMapping& operator=(const StoreMapping&) = default;

 public:
  StoreMapping(StoreMapping&&)            = default;
  StoreMapping& operator=(StoreMapping&&) = default;

 public:
  bool for_future() const;
  bool for_unbound_store() const;
  const Store& store() const;

 public:
  uint32_t requirement_index() const;
  std::set<uint32_t> requirement_indices() const;
  std::set<const Legion::RegionRequirement*> requirements() const;
  bool has_group() const { return group_id_ != kNoGroup; }
  uint32_t group_id() const { return group_id_; }

 public:
  void populate_layout_constraints(Legion::LayoutConstraintSet& layout_constraints) const;

 public:
  static StoreMapping default_mapping(const Store& store, StoreTarget target, bool exact = false);
  static StoreMapping default_mapping(const Store& store, const StoreMappingConfig& config);

 private:
  uint32_t group_id_;
};

struct LegateMapper {
  virtual bool is_pure() const                                                              = 0;
  virtual TaskTarget task_target(const Task& task, const std::vector<TaskTarget>& options)  = 0;
  virtual std::vector<StoreMapping> store_mappings(const Task& task,
                                                   const std::vector<StoreTarget>& options) = 0;
  virtual Scalar tunable_value(TunableID tunable_id)                                        = 0;
};

struct Debug {};
extern Debug detail_debug;

template <class T>
Debug& operator<<(Debug& debug, const T& t)
{
#if 0
  std::cout << t;
#endif
  return debug;
}

static inline Debug& operator<<(Debug& debug, std::ostream& (*pf)(std::ostream&))
{
#if 0
  std::cout << pf;
#endif
  return debug;
}

}  // namespace mapping
}  // namespace legate
