import os
from openai import OpenAI
from pyexpat.errors import messages
from tools import tool

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-e5399238a40c4729bf7c5ecf68240c9e",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

my_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": input("请输入你的问题：")},
]

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-plus",
    messages=my_messages,
    extra_body={
        "enable_thinking": True,
        "enable_search": True,
    },
    tools=tool,
    parallel_tool_calls=True,
)

reasoning_content = ""
answer_content = ""
tool_info = []
is_answering = False
print("=" * 20 + "思考过程" + "=" * 20)

# 修复：处理可能的元组响应
for chunk in completion:
    # 检查chunk是否是元组，如果是，则取第一个元素
    if isinstance(chunk, tuple):
        chunk = chunk[0]

    # 检查chunk是否有choices属性
    if hasattr(chunk, 'choices') and chunk.choices:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            reasoning_content = delta.reasoning_content
            print(reasoning_content, end="", flush=True)
        else:
            if not is_answering:
                is_answering = True
                print("\n" + "=" * 20 + "回复内容" + "=" * 20)
            if delta.content is not None:
                answer_content = delta.content
                print(answer_content, end="", flush=True)
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    index = tool_call.index
                    while len(tool_info) <= index:
                        tool_info.append({})
                    if tool_call.id:
                        tool_info[index]["id"] = tool_info[index].get("id", '') + tool_call.id
                    if tool_call.function and tool_call.function.name:
                        tool_info[index]["name"] = tool_info[index].get("name", '') + tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tool_info[index]["arguments"] = tool_info[index].get("arguments",
                                                                             '') + tool_call.function.arguments
    elif hasattr(chunk, 'usage'):
        print("\n" + "=" * 20 + "usage" + "=" * 20)
        print(chunk.usage)

print(f"\n" + "=" * 20 + "工具调用" + "=" * 20)
if not tool_info:
    print("无工具调用")
else:
    print(tool_info)

# 检查completion是否是元组，如果是，则取第一个元素
if isinstance(completion, tuple):
    completion_to_print = completion[0]
else:
    completion_to_print = completion

print(completion_to_print.model_dump_json())