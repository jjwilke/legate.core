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

#include "core/type/type_traits.h"
#include "core/utilities/span.h"
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition for legate::Scalar
 */

namespace legate {

class BufferBuilder;

/**
 * @ingroup data
 * @brief A type-erased container for scalars
 *
 * A Scalar can be owned or shared, depending on whether it owns the backing allocation:
 * If a `Scalar` is shared, it does not own the allocation and any of its copies are also
 * shared. If a `Scalar` is owned, it owns the backing allocation and releases it upon
 * destruction. Any copy of an owned `Scalar` is owned as well.
 */
class Scalar {
 public:
  Scalar(const Scalar& other);
  Scalar(Scalar&& other);

  /**
   * @brief Creates a shared `Scalar` with an existing allocation. The caller is responsible
   * for passing in a sufficiently big allocation. When `copy` is true, the scalar copies the
   * data stored in the allocation and becomes owned.
   *
   * @param type Type of the scalar(s)
   * @param data Allocation containing the data.
   */
  Scalar(std::unique_ptr<Type> type, const void* data, bool copy = false);
  ~Scalar();

 public:
  /**
   * @brief Creates an owned scalar from a scalar value
   *
   * @tparam T The scalar type to wrap
   *
   * @param value A scalar value to create a `Scalar` with
   */
  template <typename T>
  Scalar(T value);
  /**
   * @brief Creates an owned scalar of a specified type from a scalar value
   *
   * @tparam T The scalar type to wrap
   *
   * @param type The type of the scalar
   * @param value A scalar value to create a `Scalar` with
   */
  template <typename T>
  Scalar(T value, std::unique_ptr<Type> type);
  /**
   * @brief Creates an owned scalar from a string. The value from the
   * original string will be copied.
   *
   * @param string A string to create a `Scalar` with
   */
  Scalar(const std::string& string);

  /**
   * @brief Creates an owned scalar from a tuple of scalars. The values in the input vector
   * will be copied.
   *
   * @param values A vector that contains elements of a tuple
   */
  template <typename T>
  Scalar(const std::vector<T>& values);

 public:
  /**
   * @brief Creates a point scalar
   *
   * @param point A point from which the scalar should be constructed
   */
  template <int32_t DIM>
  Scalar(const Point<DIM>& point);
  /**
   * @brief Creates a rect scalar
   *
   * @param point A rect from which the scalar should be constructed
   */
  template <int32_t DIM>
  Scalar(const Rect<DIM>& rect);

 public:
  Scalar& operator=(const Scalar& other);

 private:
  void copy(const Scalar& other);

 public:
  /**
   * @brief Returns the data type of the scalar
   *
   * @return Data type
   */
  const Type& type() const { return *type_; }
  /**
   * @brief Returns the size of allocation for the `Scalar`.
   *
   * @return The size of allocation
   */
  size_t size() const;

 public:
  /**
   * @brief Returns the value stored in the `Scalar`. The call does no type checking;
   * i.e., passing a wrong type parameter will not be caught by the call.
   *
   * @tparam VAL Type of the value to unwrap
   *
   * @return The value stored in the `Scalar`
   */
  template <typename VAL>
  VAL value() const;
  /**
   * @brief Returns values stored in the `Scalar`. If the `Scalar` contains a scalar,
   * a unit span will be returned.
   *
   * @return Values stored in the `Scalar`
   */
  template <typename VAL>
  Span<const VAL> values() const;
  /**
   * @brief Returns a raw pointer to the backing allocation
   *
   * @return A raw pointer to the `Scalar`'s data
   */
  const void* ptr() const { return data_; }

 public:
  /**
   * @brief Serializes the scalar into a buffer
   *
   * @param buffer A BufferBuilder into which the scalar needs to be serialized
   */
  void pack(BufferBuilder& buffer) const;

 private:
  bool own_{false};
  std::unique_ptr<Type> type_{nullptr};
  const void* data_;
};

}  // namespace legate

#include "core/data/scalar.inl"
