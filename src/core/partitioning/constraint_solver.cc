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

#include <sstream>

#include "legion.h"

#include "core/data/detail/logical_store.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_solver.h"
#include "core/runtime/operation.h"

namespace legate {
extern Legion::Logger log_legate;
}  // namespace legate

namespace legate::detail {

namespace {

struct UnionFindEntry {
  UnionFindEntry(const Variable* symb, const detail::LogicalStore* store)
    : partition_symbol(symb), restrictions(store->compute_restrictions()), next(nullptr), size(1)
  {
  }

  UnionFindEntry* unify(UnionFindEntry* other)
  {
    UnionFindEntry* self = this;
    if (self->size < other->size) std::swap(self, other);

    auto end = self;
    while (end->next != nullptr) end = end->next;
    end->next = other;
    end->size += other->size;
    return self;
  }

  const Variable* partition_symbol;
  Restrictions restrictions;
  UnionFindEntry* next;
  size_t size;
};

}  // namespace

struct ConstraintSolver::EquivClass {
  EquivClass(const UnionFindEntry* entry)
  {
    partition_symbols.reserve(entry->size);
    partition_symbols.push_back(entry->partition_symbol);
    restrictions = entry->restrictions;

    auto* next = entry->next;
    while (next != nullptr) {
      partition_symbols.push_back(next->partition_symbol);
      join_inplace(restrictions, next->restrictions);
      next = next->next;
    }
  }

  std::vector<const Variable*> partition_symbols;
  Restrictions restrictions;
};

ConstraintSolver::ConstraintSolver() {}

ConstraintSolver::~ConstraintSolver() {}

void ConstraintSolver::add_partition_symbol(const Variable* partition_symbol)
{
  partition_symbols_.insert(partition_symbol);
}

void ConstraintSolver::add_constraint(const Constraint* constraint)
{
  constraints_.push_back(constraint);
}

void ConstraintSolver::solve_constraints()
{
  std::vector<UnionFindEntry> entries;
  std::map<const Variable, UnionFindEntry*> table;

  // Initialize the table by creating singleton equivalence classes
  const auto& all_symbols = partition_symbols();
  entries.reserve(all_symbols.size());

  auto initialize = [&entries, &table](const auto& all_symbols) {
    for (auto& symb : all_symbols) {
      // TODO: partition symbols can be independent of any stores of the operation
      //       (e.g., when a symbol subsumes a union of two other symbols)
      auto* store = symb->operation()->find_store(symb);
      entries.emplace_back(symb, store);
      table.insert({*symb, &entries.back()});
    }
  };
  initialize(all_symbols);

  // Unify equivalence classes based on alignment constraints
  auto handle_alignment = [&table](const Alignment* alignment) {
    auto update_table = [&table](UnionFindEntry* old_cls, UnionFindEntry* new_cls) {
      while (old_cls != nullptr) {
        table[*old_cls->partition_symbol] = new_cls;
        old_cls                           = old_cls->next;
      }
    };

    std::vector<const Variable*> part_symbs_to_unify;
    alignment->find_partition_symbols(part_symbs_to_unify);
#ifdef DEBUG_LEGATE
    assert(!part_symbs_to_unify.empty());
#endif

    auto it           = part_symbs_to_unify.begin();
    auto* equiv_class = table[**it++];
#ifdef DEBUG_LEGATE
    assert(equiv_class != nullptr);
#endif
    for (; it != part_symbs_to_unify.end(); ++it) {
      auto* to_unify = table[**it];
      auto* result   = equiv_class->unify(to_unify);

      if (result != equiv_class) update_table(equiv_class, result);
      if (result != to_unify) update_table(to_unify, result);
    }
  };

  // Set the restrictions according to broadcasting constraints
  auto handle_broadcast = [&table](const Broadcast* broadcast) {
    auto* variable    = broadcast->variable();
    auto& axes        = broadcast->axes();
    auto* equiv_class = table.at(*variable);
    for (uint32_t idx = 0; idx < axes.size(); ++idx) {
      uint32_t axis = axes[idx];
      // TODO: We want to check the axis eagerly and raise an exception
      // if it is out of bounds
      if (axis >= equiv_class->restrictions.size()) continue;
      equiv_class->restrictions[axes[idx]] = Restriction::FORBID;
    }
  };

  // Reflect each constraint to the solver state
  for (auto& constraint : constraints_) switch (constraint->kind()) {
      case Constraint::Kind::ALIGNMENT: {
        handle_alignment(constraint->as_alignment());
        break;
      }
      case Constraint::Kind::BROADCAST: {
        handle_broadcast(constraint->as_broadcast());
        break;
      }
    }

  // Combine states of each union of equivalence classes into one
  std::unordered_set<UnionFindEntry*> distinct_entries;
  for (auto& [_, entry] : table) distinct_entries.insert(entry);

  for (auto* entry : distinct_entries) {
    auto equiv_class = std::make_unique<EquivClass>(entry);
    for (auto* symb : equiv_class->partition_symbols)
      equiv_class_map_.insert({*symb, equiv_class.get()});
    equiv_classes_.push_back(std::move(equiv_class));
  }
}

const std::vector<const Variable*>& ConstraintSolver::find_equivalence_class(
  const Variable* partition_symbol) const
{
  return equiv_class_map_.at(*partition_symbol)->partition_symbols;
}

const Restrictions& ConstraintSolver::find_restrictions(const Variable* partition_symbol) const
{
  return equiv_class_map_.at(*partition_symbol)->restrictions;
}

void ConstraintSolver::dump()
{
  log_legate.debug("===== Constraint Graph =====");
  log_legate.debug() << "Variables:";
  for (auto& symbol : partition_symbols_.elements())
    log_legate.debug() << "  " << symbol->to_string();
  log_legate.debug() << "Constraints:";
  for (auto& constraint : constraints_) log_legate.debug() << "  " << constraint->to_string();
  log_legate.debug("============================");
}

const std::vector<const Variable*>& ConstraintSolver::partition_symbols() const
{
  return partition_symbols_.elements();
}

const std::vector<const Constraint*>& ConstraintSolver::constraints() const { return constraints_; }

}  // namespace legate::detail
