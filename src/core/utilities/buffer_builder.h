/* Copyright 2021-2023 NVIDIA Corporation
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

#include "legion.h"

#include "core/utilities/tuple.h"

namespace legate {

class BufferBuilder {
 public:
  BufferBuilder();

 public:
  template <typename T>
  void pack(const T& value);
  template <typename T>
  void pack(const std::vector<T>& values);
  template <typename T>
  void pack(const tuple<T>& values);
  void pack_buffer(const void* buffer, size_t size);

 public:
  Legion::UntypedBuffer to_legion_buffer() const;

 private:
  std::vector<int8_t> buffer_;
};

}  // namespace legate

#include "core/utilities/buffer_builder.inl"
