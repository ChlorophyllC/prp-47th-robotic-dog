from openai import OpenAI
from typing import List,Tuple
import os
import ast
import algorithms
import certifi
# 设置环境变量指向正确的证书文件
os.environ['SSL_CERT_FILE'] = certifi.where()

def call_LLM(vehicles:List[List[Tuple[float, float]]],
             destinations:List[List[Tuple[float, float]]],
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
        api_key = "",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    explain_path = "algorithms_readme_for_LLM.md"
    with open(explain_path, "r",encoding="utf-8") as f:
        code_explain = f.read()

    vehicle_data="小车初始位置为："
    destination_data="目的地位置为："
    index=0
    for vehicle in vehicles:
        vehicle_data=vehicle_data+"小车"+str(index)+"："+str(vehicle)+","
        index+=1

    index=0
    for destination in destinations:
        destination_data=destination_data+"目的地"+str(index)+"："+str(destination)+","
        index+=1

    # 创建聊天完成请求
    completion = client.chat.completions.create(
        model="qwen-turbo-latest", 
        messages=[
            {"role": "user", "content": f"现在有以下函数声明，利用这些函数完成给定任务：\n```python\n{code_explain}\n```"},
            {"role": "user", "content": "初始参数：" + vehicle_data + destination_data},
            {"role": "user", "content": "命令："+command},
            {"role": "user", "content": "利用所给函数完成任务，按顺序依次输出调用函数，不需要补充函数体，也不要进行解释。"}
        ],
    )
    if completion.choices[0] is not None:
        function_list=ast.literal_eval(completion.choices[0].message.content)
        return function_list
    return None

def Interpret_function_list(function_list:List, obj):
    """
    释读函数列表并依次执行

    Args:
        function_list:大模型给出的函数调用列表
        obj: 包含待调用方法的对象
    """
    results = []
    for fun in function_list:
        try:
            # 构造方法调用字符串并执行
            method_call = f"obj.{fun}"
            result = eval(method_call)
            
            # 确保每个结果都是字典形式
            if not isinstance(result, dict):
                result = {"result": result}
                
            results.append(result)
        except Exception as e:
            # 如果调用失败，记录错误信息
            results.append({"error": str(e), "failed_function": fun})
    
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
    function_list=call_LLM(vehicles,destinations,command)
    if function_list is None:
        print("Error: LLM failed!")
    else:
        print("结果为：")
        print(Interpret_function_list(function_list, obj))