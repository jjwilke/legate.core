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

#pragma once

#include <map>
#include <memory>
#include <vector>

namespace legate {

struct Constraint;
struct Variable;

struct ConstraintGraph {
 public:
  void add_variable(std::shared_ptr<Variable> variable);
  void add_constraint(std::shared_ptr<Constraint> constraint);

 public:
  void join(const ConstraintGraph& other);

 public:
  void dump();

 public:
  const std::vector<std::shared_ptr<Variable>>& variables() const;
  const std::vector<std::shared_ptr<Constraint>>& constraints() const;

 private:
  std::vector<std::shared_ptr<Variable>> variables_;
  std::vector<std::shared_ptr<Constraint>> constraints_;
};

}  // namespace legate
