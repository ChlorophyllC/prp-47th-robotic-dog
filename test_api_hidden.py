from openai import OpenAI
from typing import List
import os
import json
import algorithms
import certifi
# 设置环境变量指向正确的证书文件
os.environ['SSL_CERT_FILE'] = certifi.where()

def call_LLM(vehicle_num:int,
             destination_num:int,
             command:str):
    """
    调用大模型理解指令、规划任务

    Args:
        vehicles: 所有车辆的矩形定义，每个车辆由4个角点组成
        destinations: 所有目的地的矩形定义，每个目的地由4个角点组成
        command: 输入的指令

    Return:
        返回一个函数调用的列表。若出错则返回None
    """
    # 初始化OpenAI客户端
    client = OpenAI(
        # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
        api_key = os.getenv("YOUR_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    explain_path = "algorithms_readme_for_LLM.json"
    with open(explain_path, "r",encoding="utf-8") as f:
        tools = json.load(f)

    # 创建聊天完成请求
    completion = client.chat.completions.create(
        model="qwen3.5-plus", 
        messages=[
            {"role": "user", "content": f"利用外部工具完成给定任务。"},
            {"role": "user", "content": f"现有{vehicle_num}辆小车和{destination_num}个目的地，小车索引为0到{vehicle_num-1}，目的地索引为0到{destination_num-1}。"},
            {"role": "user", "content": "命令："+command}
        ],
        tools=tools,
        extra_body={"enable_thinking": False}
    )
    
    response = completion.choices[0].message
    if response.tool_calls is None:
        return None
    else:
        function_list = []
        for tool_call in response.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            function_list.append({"name": function_name, "arguments": arguments})
        return function_list

def Interpret_function_list(function_list:List, obj):
    """
    释读函数列表并依次执行

    Args:
        function_list:大模型给出的函数调用列表
        obj: 包含待调用方法的对象
    """
    results = []
    for func in function_list:
        func_name = func["name"]
        arguments = func["arguments"]
        try:
            call = getattr(obj,func_name)
            result = call(**arguments)

            # 确保每个结果都是字典形式
            if not isinstance(result, dict):
                result = {"result": result}

            results.append(result)

        except AttributeError:
            # 如果调用失败，记录错误信息
            results.append({"error": "function not found", "failed_function": func})

    return results

if __name__=="__main__":
    vehicles=[
        [[3, 18], [3, 16], [5, 16], [5, 18]],
        [[9, 18], [9, 17], [10, 17], [10, 18]],
        [[3, 8], [3, 9], [4, 9], [4, 8]],
        [[17, 15], [17, 14], [18, 14], [18, 15]]
    ]

    obstacles =[[[8, 14], [8, 13], [10, 13], [10, 14]],
    [[11, 16], [11, 12], [12, 12], [12, 16]],
    [[3, 12], [3, 11], [5, 11], [5, 12]]]

    destinations=[
        [[11, 8], [11, 5], [13, 5], [13, 8]],
        [[3, 5], [3, 3], [5, 3], [5, 5]]
    ]
    command="让车辆0，1包围目的地1,再让车辆0到达目的地0"
    obj=algorithms.PathPlanner(vehicles,obstacles,destinations)

    function_list=call_LLM(len(vehicles),len(destinations),command)

    if function_list is None:
        print("Error: LLM failed!")
    else:
        print("结果为：")
        print(Interpret_function_list(function_list, obj))
