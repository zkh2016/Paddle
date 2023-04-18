#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.incubate.nn import memory_efficient_attention

from .flash_attention import flash_attention


def scaled_dot_product_attention(
    query,
    key,
    value,
    dropout_rate=0.0,
    is_causal=False,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API is only support inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        dropout_rate(float): The dropout ratio.
        is_causal(bool): Whether enable causal mode.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.

    Examples:
        .. code-block:: python

            # required: skiptest
            import paddle

            q = paddle.rand((1, 128, 2, 16), dtype=paddle.float16)

            output = paddle.nn.functional.scaled_dot_product_attention(q, q, q, 0.9, False)
            print(output)
    """
    """
    if in_dynamic_mode():
        out = _C_ops.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_rate,
            is_causal
        )
        return out

    helper = LayerHelper('scaled_dot_product_attention', **locals())
    dtype = helper.input_dtype(input_param_name='query')
    out = helper.create_variable_for_type_inference(dtype)
    softmax = helper.create_variable_for_type_inference(dtype)
    softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
    seed_offset = helper.create_variable_for_type_inference(paddle.int64)
    inputs = {
        'query': query,
        'key': key,
        'value': value,
    }
    outputs = {
        'out': out,
        'softmax': softmax,
        'softmax_lse': softmax_lse,
        'seed_offset': seed_offset,
    }
    helper.append_op(
        type='scaled_dot_product_attention',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'dropout_rate': dropout_rate,
            'is_causal': is_causal,
        },
    )
    return out
    """

    head_dim = query[3]
    if head_dim < 128:
        return flash_attention(
            query,
            key,
            value,
            dropout=dropout_rate,
            causal=is_causal,
            return_softmax=False,
            training=True,
            name=name,
        )
    else:
        return memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=None,
            p=dropout_rate,
            scale=None,
            training=True,
        )
