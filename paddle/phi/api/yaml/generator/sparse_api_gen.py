# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import argparse
import re

from api_gen import ForwardAPI


class SparseAPI(ForwardAPI):

    def __init__(self, api_item_yaml):
        super(SparseAPI, self).__init__(api_item_yaml)

    def gene_api_declaration(self):
        return f"""
// {", ".join(self.outputs['names'])}
{super(SparseAPI, self).gene_api_declaration()}
"""

    def gene_output(self,
                    out_dtype_list,
                    out_tensor_type_list=None,
                    code_indent='',
                    inplace_flag=False):
        kernel_output = []
        output_names = []
        output_create = ""
        return_type = self.get_return_type_with_intermediate(inplace_flag)
        output_type_map = {
            'dense': 'TensorType::DENSE_TENSOR',
            'sparse_coo': 'TensorType::SPARSE_COO',
            'sparse_csr': 'TensorType::SPARSE_CSR'
        }

        if len(out_dtype_list) == 1:
            kernel_output.append('kernel_out')
            output_names.append('kernel_out')
            inplace_assign = " = " + self.inplace_map[self.outputs['names'][
                0]] if inplace_flag and self.inplace_map is not None and self.outputs[
                    'names'][0] in self.inplace_map else ""
            output_create = f"""
    {return_type} api_output{inplace_assign};
    auto* kernel_out = SetSparseKernelOutput(&api_output, {output_type_map[out_dtype_list[0]]});"""

        elif len(out_dtype_list) > 1:
            output_create = f"""
    {return_type} api_output;"""

            if inplace_flag:
                output_create = f"""
    {return_type} api_output{{"""

                for out_name in self.outputs['names']:
                    if out_name in self.inplace_map:
                        output_create = output_create + self.inplace_map[
                            out_name] + ', '
                    else:
                        output_create += 'Tensor(), '
                output_create = output_create[:-2] + '};'

            for i in range(len(out_dtype_list)):
                kernel_output.append(f'kernel_out_{i}')
                output_names.append(f'kernel_out_{i}')
                output_create = output_create + f"""
    auto* kernel_out_{i} = SetSparseKernelOutput(&std::get<{i}>(api_output), {output_type_map[out_dtype_list[i]]});"""

        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))

        return kernel_output, output_names, output_create

    def gen_sparse_kernel_context(self, kernel_output_names):
        input_trans_map = {
            'const Tensor&':
            'const phi::TenseBase&',
            'const std::vector<Tensor>&':
            'const std::vector<phi::TenseBase>&',
            'const paddle::optional<Tensor>&':
            'paddle::optional<const phi::TenseBase&>'
        }
        out_trans_map = {
            'Tensor': 'phi::TenseBase*',
            'std::vector<Tensor>': 'std::vector<phi::TenseBase*>'
        }
        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']

        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        kernel_context_code = ""
        for param in kernel_param:
            if param in input_names:
                if param in self.optional_vars:
                    kernel_context_code = kernel_context_code + f"""
    kernel_context.EmplaceBackInput({param} ? {param}->impl().get() : nullptr);"""
                else:
                    kernel_context_code = kernel_context_code + f"""
    kernel_context.EmplaceBackInput({param}.impl().get());"""

                continue
            if param in attr_names:
                # set attr for kernel_context
                if 'IntArray' in self.attrs['attr_info'][param][0]:
                    param = 'phi::IntArray(' + param + ')'
                elif 'Scalar' in self.attrs['attr_info'][param][0]:
                    param = 'phi::Scalar(' + param + ')'
            elif isinstance(param, bool):
                param = str(param).lower()
            else:
                param + str(param) + ", "
            kernel_context_code = kernel_context_code + f"""
    kernel_context.EmplaceBackAttr({param});"""

        for out_name in kernel_output_names:
            kernel_context_code = kernel_context_code + f"""
    kernel_context.EmplaceBackOutput({out_name});"""

        return kernel_context_code

    def gene_infer_meta(self, kernel_output_names, code_indent) -> str:
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        if not hasattr(self, 'infer_meta'):
            return ''
        infer_meta = self.infer_meta

        infer_meta_params = infer_meta['param'] if infer_meta[
            'param'] is not None else input_names + attr_names
        # generate meta tensors
        meta_tensor_code = ""
        param_code = ""
        for param in infer_meta_params:
            if param in input_names:
                if self.inputs['input_info'][param] == "const Tensor&":
                    param_code = param_code + "MakeMetaTensor(*" + param + ".impl()), "
                elif self.inputs['input_info'][
                        param] == "const std::vector<Tensor>&":
                    meta_tensor_code = meta_tensor_code + f"""
{code_indent}  auto {param}_meta_vec = MakeMetaTensor(*{param}.impl());
{code_indent}  std::vector<const phi::MetaTensor*> {param}_metas({param}_meta_vec.size());
{code_indent}  for (size_t i = 0; i < {param}_meta_vec.size(); ++i) {{
{code_indent}    {param}_metas[i] = &{param}_meta_vec[i];
{code_indent}  }}
"""
                    param_code = param_code + param + "_metas, "
                elif self.inputs['input_info'][
                        param] == "const paddle::optional<std::vector<Tensor>>&":
                    meta_tensor_code = meta_tensor_code + f"""
{code_indent}  auto {param}_meta_vec = MakeMetaTensor(*{param}.impl());
{code_indent}  paddle::optional<std::vector<const phi::MetaTensor*>> {param}_metas({param}_meta_vec.size());
{code_indent}  for (size_t i = 0; i < {param}_meta_vec.size(); ++i) {{
{code_indent}    {param}_metas->at(i) = &{param}_meta_vec[i];
{code_indent}  }}
"""
                    param_code = param_code + param + "_metas, "
                elif param in self.optional_vars:
                    param_code = param_code + "MakeMetaTensor(*" + param + ".impl()), "
                else:
                    raise ValueError(
                        f"{self.api} : Param of infer_meta error : {self.inputs['input_info'][param]} type is not supported."
                    )
            elif param in attr_names:
                param_code = param_code + param + ", "
            elif isinstance(param, str):
                param_code = param_code + "\"" + param + "\", "
            elif isinstance(param, bool):
                param_code = param_code + str(param).lower() + ", "
            else:
                param_code = param_code + str(param) + ", "

        PREFIX_META_TENSOR_NAME = 'meta_'

        for i, out_name in enumerate(kernel_output_names):
            if self.outputs['types'][i] == 'std::vector<Tensor>':
                meta_tensor_code = meta_tensor_code + f"""
{code_indent}  auto {out_name}_{PREFIX_META_TENSOR_NAME}vec = MakeMetaTensor({out_name});
{code_indent}  std::vector<phi::MetaTensor*> {out_name}_metas({out_name}_{PREFIX_META_TENSOR_NAME}vec.size());
{code_indent}  for (size_t i = 0; i < {out_name}_{PREFIX_META_TENSOR_NAME}vec.size(); ++i) {{
{code_indent}    {out_name}_metas[i] = {out_name}[i] ? &{out_name}_{PREFIX_META_TENSOR_NAME}vec[i] : nullptr;
{code_indent}  }}"""

                param_code = param_code + out_name + '_metas, '
            else:
                meta_tensor_code = meta_tensor_code + code_indent + "  phi::MetaTensor " + out_name.replace(
                    'kernel_',
                    PREFIX_META_TENSOR_NAME) + "(" + out_name + ");\n"
                if len(kernel_output_names) == 1:
                    param_code = param_code + f"&{out_name.replace('kernel_', PREFIX_META_TENSOR_NAME)}, "
                else:
                    param_code = param_code + f"{out_name} ? &{out_name.replace('kernel_', PREFIX_META_TENSOR_NAME)} : nullptr, "

        param_code = param_code[:-2]
        return f"""{meta_tensor_code}
{code_indent}  phi::sparse::{infer_meta['func']}({param_code});
"""

    def gen_sparse_kernel_code(self, kernel_name, inplace_flag=False):
        _, kernel_output_names, output_create = self.gene_output(
            self.kernel['dispatch'][kernel_name][1], None, '', inplace_flag)

        kernel_context_code = self.gen_sparse_kernel_context(
            kernel_output_names)
        return_code = "" if len(
            self.gene_return_code()) == 0 else "  " + self.gene_return_code()
        return f"""
    VLOG(6) << "{self.api} api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "{kernel_name}", {{kernel_backend, kernel_layout, kernel_data_type}});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "{self.api} api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);
{output_create}
{self.gene_infer_meta(kernel_output_names, '')}
{kernel_context_code}
    phi_kernel(&kernel_context);
  {return_code}"""

    def get_condition_code(self, kernel_name):
        assert self.kernel['dispatch'][kernel_name], \
                f"{self.api} api: the tensor type of inputs and outputs for kernel isn't set, see also 'kernel:func' of 'conv3d' in sparse_api.yaml."
        input_types = self.kernel['dispatch'][kernel_name][0]
        sparse_type_map = {
            'sparse_coo': 'DataLayout::SPARSE_COO',
            'sparse_csr': 'DataLayout::SPARSE_CSR'
        }
        condition_list = []
        for i, in_type in enumerate(input_types):
            if in_type == "dense":
                if self.inputs['names'][i] in self.optional_vars:
                    condition_list.append(
                        f"(!{self.inputs['names'][i]} || phi::DenseTensor::classof({self.inputs['names'][i]}->impl().get()))"
                    )
                else:
                    condition_list.append(
                        f"phi::DenseTensor::classof({self.inputs['names'][i]}.impl().get())"
                    )
            else:
                if in_type == 'sparse_coo':
                    condition_list.append(
                        f"{self.inputs['names'][i]}.is_sparse_coo_tensor()")
                else:
                    condition_list.append(
                        f"{self.inputs['names'][i]}.is_sparse_csr_tensor()")

        return " && ".join(condition_list)

    def gene_dispatch_code(self, kernel_name, inplace_flag=False):
        return f"""
  if ({self.get_condition_code(kernel_name)}) {{
{self.gen_sparse_kernel_code(kernel_name, inplace_flag)}
  }}
"""

    def gene_base_api_code(self, inplace_flag=False):
        api_func_name = self.get_api_func_name()
        if inplace_flag and api_func_name[-1] != '_':
            api_func_name += '_'
        kernel_dispatch_code = f"{self.gene_kernel_select()}\n"
        for kernel_name in self.kernel['func']:
            kernel_dispatch_code += self.gene_dispatch_code(
                kernel_name, inplace_flag)

        return f"""
PADDLE_API {self.get_return_type()} {api_func_name}({self.get_define_args()}) {{
{kernel_dispatch_code}
  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of ({self.api}) for input tensors is unimplemented, please check the type of input tensors."));
}}
"""


def header_include():
    return """
#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/sparse_api_custom_impl.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/infermeta/sparse/binary.h"
#include "paddle/phi/infermeta/sparse/ternary.h"
"""


def api_namespace():
    return ("""
namespace paddle {
namespace experimental {
namespace sparse {

""", """

}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
""")


def generate_api(api_yaml_path, header_file_path, source_file_path):

    with open(api_yaml_path, 'r') as f:
        apis = yaml.load(f, Loader=yaml.FullLoader)
    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = "paddle/phi/api/include/sparse_api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for api in apis:
        sparse_api = SparseAPI(api)
        if sparse_api.is_dygraph_api:
            sparse_api.is_dygraph_api = False
        header_file.write(sparse_api.gene_api_declaration())
        source_file.write(sparse_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ Sparse API files')
    parser.add_argument('--api_yaml_path',
                        help='path to sparse api yaml file',
                        default='paddle/phi/api/yaml/sparse_api.yaml')

    parser.add_argument('--api_header_path',
                        help='output of generated api header code file',
                        default='paddle/phi/api/include/sparse_api.h')

    parser.add_argument('--api_source_path',
                        help='output of generated api source code file',
                        default='paddle/phi/api/lib/sparse_api.cc')

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(api_yaml_path, header_file_path, source_file_path)


if __name__ == '__main__':
    main()
