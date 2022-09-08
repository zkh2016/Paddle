/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {
namespace sparse {
// Common InferMeta Functions of SparseTensor for unary operators:
void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out);
void SparseCooTensorInferMeta(const MetaTensor& values,
                              const MetaTensor& indices,
                              const IntArray& dense_shape,
                              MetaTensor* out);

void ValuesInferMeta(const MetaTensor& x, MetaTensor* out);

void IndicesInferMeta(const MetaTensor& x, MetaTensor* out);

}  // namespace sparse
}  // namespace phi
