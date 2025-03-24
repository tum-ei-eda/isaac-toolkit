import re
import subprocess
import os
import argparse

def process_sub_funcs_file(input_file_path):
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()
    parent_match = None
    dic = {}
    sub_function_pattern = re.compile(r"(\d{1,3}(?:,\d{3})*)\s*\((\d+\.\d+%)\)\s*=>\s*([^\s:]+:[a-zA-Z0-9_]+)\s*\((\d+x)\)")
    

    parent_pattern = re.compile(r"(\d{1,3}(?:,\d{3})*)\s*\(\s*(\d+\.\d+%)\s*\)\s*TVM_DLL\s+int32_t\s+(\w+)\s*\((.*?)\)\s*\{")

    for line in lines:
        line = line.strip()
        if parent_match is None:
            parent_match = parent_pattern.match(line)
        else:
            sub_match = sub_function_pattern.match(line)
            if sub_match:
                parent_name = parent_match.group(3)
                file_function = sub_match.group(3)
                _, sub_name = file_function.split(':')
                cost = sub_match.group(1)
                dic[parent_name] = (sub_name, cost)
                parent_match = None
                sub_match = None
    print(dic)
    return dic

def process_main_func_file(input_file_path, output_file_path, info):
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()
    function_stack = []  # Stack to handle nested functions
    result = []
    total = 0
    parent_match = None

    number_pattern = re.compile(r"^\s*(\d{1,3}(?:,\d{3})*)")
    function_pattern = re.compile(r"(\d{1,3}(?:,\d{3})*)\s*\((\d+\.\d+%)\)\s*=>\s*([^\s:]+:[a-zA-Z0-9_]+)\s*\((\d+x)\)")
    parent_pattern = re.compile(r"(\d{1,3}(?:,\d{3})*)\s*\(\s*(\d+\.\d+%)\s*\)\s*TVM_DLL\s+int32_t\s+tvmgen_default___tvm_main__\s*\((.*?)\)\s*\{")

    for line in lines:
        line = line.strip()
        
        if parent_match is None:
            parent_match = parent_pattern.match(line)
        number_match = number_pattern.match(line)
        
        if number_match:
            num = int(number_match.group(1).replace(",", "")) 
            total += num
    print(total)
    if parent_match:
        print(line)
        print(parent_match)
        function_name = "tvmgen_default___tvm_main__"
        formatted = f"{function_name} <int32_t {function_name} () takes {total} cycles>:"
        
        result.append(f"{formatted}")
        
        function_stack.append(function_name)    

    for line in lines:
        line = line.strip()

        match = function_pattern.match(line)
        if match:
            print(line)
            print(match)
            file_function = match.group(3)
            file_path, function_name = file_function.split(':')
            cost = match.group(1)  # This contains both the cost in milliseconds and percentage
            
            cost_ms = cost.split(" ")[0]
            formatted = f"{function_name} <int32_t {function_name} () takes {cost_ms} cycles>"
            sub_formatted = f"{info[function_name][0]} <void {info[function_name][0]} () takes {info[function_name][1]} cycles>"
            if function_stack:
                result[-1] += f"\n    {formatted}:\n        {sub_formatted}"
                
            else:
                result.append(f"{formatted}")
            
            function_stack.append(function_name)

        if line == "}":
            function_stack.pop()
    
    with open(output_file_path, 'w') as outfile:
        outfile.write("\n".join(result))
        outfile.write("\n")
    
    print(f"Processed content written to {output_file_path}")


def split_file_content(file1, file2, file3):
    if not os.path.exists(file1):
        print(f"Error: {file1} does not exist.")
        return
    
    start_marker = "ob=/nas/ei/share/TUEIEDAscratch/ge74mos/ecomai_perf_analyzer/isaac-toolkit/examples/mlonmcu/sess/elf/generic_mlonmcu"
    lib2_start_marker = "fl=/nas/ei/share/TUEIEDAscratch/ge74mos/ecomai_perf_analyzer/isaac-toolkit/workspace/temp/sessions/10/runs/0/codegen/host/src/default_lib2.c"
    end_marker = "fl=???"
    lib1_start_marker = "fl=/nas/ei/share/TUEIEDAscratch/ge74mos/ecomai_perf_analyzer/isaac-toolkit/workspace/temp/sessions/10/runs/0/codegen/host/src/default_lib1.c"
    
    with open(file1, 'r') as f:
        lines = f.readlines()
    
    common_prefix = []
    lib2_content = []
    lib1_content = []
    
    # Extract the common prefix
    found_start_marker = False
    for line in lines:
        common_prefix.append(line)
        if start_marker in line:
            found_start_marker = True
            break
    
    if not found_start_marker:
        print("Error: Start marker not found in file1.")
        return
    
    lib2_marker = False
    lib1_marker = False
    
    for line in lines[len(common_prefix):]:
        if lib2_start_marker in line:
            lib2_marker = True

        if lib1_start_marker in line:
            lib1_marker = True
        
        if end_marker in line:
            lib2_marker = False
        
        if lib2_marker:
            lib2_content.append(line)
        elif lib1_marker:
            lib1_content.append(line)
    
    lib1_content = common_prefix + lib1_content
    lib2_content = common_prefix + lib2_content
    
    with open(file2, 'w') as f2:
        f2.writelines(lib1_content)
    
    with open(file3, 'w') as f3:
        f3.writelines(lib2_content)
    
    print(f"Successfully split content into {file2} and {file3}.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generating a CFLOW format file including runtime information.")
    parser.add_argument("input_file", help="Path to the input file", default="callgrind_pos.out")
    parser.add_argument("output_file", help="Specify the output file for results", default="cflow_output.txt")
  
    
    args = parser.parse_args()
    lib1 = "lib1.out"
    lib2 = "lib2.out"
    split_file_content(args.input_file, lib1, lib2)

    with open("main_func.txt", "w") as main_file:
        subprocess.run(["callgrind_annotate", "lib1.out"], stdout=main_file)

    with open("sub_funcs.txt", "w") as sub_file:
        subprocess.run(["callgrind_annotate", "lib2.out"], stdout=sub_file)

    input_file = 'main_func.txt'
    info = process_sub_funcs_file('sub_funcs.txt')
    process_main_func_file(input_file, args.output_file, info)

if __name__ == "__main__":
    main()