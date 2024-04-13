# Load the stravinsky file
from compiler.utils import CompilerError

with open('sha256_full.strav', 'r') as f:
    file = f.readlines()

print(file)

# Create output file
f = open("compiled_transformer.py", "w")

g = open("constants.py", "w")

f.write("import time\nfrom tqdm import tqdm\nfrom compiler.lib import *\n\nt = time.time()\n\n")

# Get the main input
input_text = ""
for line in file:
    if "INPUT" in line:
        input_text = line.split("=")[1].strip()
        break

if input_text == "":
    raise CompilerError("No input found. The file must contain a line like `INPUT = 1101011101011010`")

g.write(f"INPUT_LENGTH = {len(input_text)}\n")
g.close()

# First, we must determine how many total registers we need
registers = ['tchaikovsky', 'anti_tchaikovsky', 'zeros', 'ones']
user_defined_constant_register_values = []

# Add constant registers
for line in file:
    if line[0] == "$":
        registers.append(line[1:].split("=")[0].strip())
        user_defined_constant_register_values.append(line[1:].split("=")[1].strip())

# Extract the lines of the file between lines containing PROGRAM_START and PROGRAM_END
program_lines = []
program_started = False

for line in file:
    if "PROGRAM_START" in line:
        program_started = True
    elif "PROGRAM_END" in line:
        program_started = False
    elif program_started:
        program_lines.append(line)

for line in program_lines:
    if "=" in line:
        rname = line.split("=")[0].strip()
        if rname not in registers:
            registers.append(rname)

print(registers)

# Get the list of tokens
# In the file, the line looks like `TOKENS = /0\1\2/`
# We need to convert this to the string `tokens = list('012')`

tokens = []

for line in file:
    if "TOKENS" in line:
        tokens = line.split("/")[1].split("/")[0].strip().split("\\")

if len(tokens) == 0:
    raise CompilerError("No tokens found. The file must contain a line like `TOKENS = /0\1\2/`")

f.write(f"tokens = {tokens}\n")
f.write("pos = Register('pos', 2)\n")

# Create the register objects
for register in registers:
    f.write(f"{register} = Register('{register}', 1)\n")

num_work_registers = -1

for line in file:
    if "NUM_WORK_REGISTERS" in line:
        num_work_registers = int(line.split("=")[1].strip())
        break

if num_work_registers == -1:
    raise CompilerError("No NUM_WORK_REGISTERS found. The file must contain a line like `NUM_WORK_REGISTERS = 5`")

# Create a healthy amount of work registers
f.write(f"""
work_registers = []
for i in range({num_work_registers}):
    work_registers.append(Register(f'work_{{i}}', len(tokens)))
""")

# Create the main embedding
# embedding = EmbeddedState(tokens, [pos, tchaikovsky, anti_tchaikovsky, zero_register, input_copy, input_copy2, shifted, shiftedl] + work_registers)

embedding_line = "embedding = EmbeddedState(tokens, [pos, "
for register in registers:
    embedding_line += f"{register}, "
embedding_line = embedding_line[:-2] + "] + work_registers)\n"

f.write(embedding_line)

# Now, create the input
# example = embedding.embed(embedding.tokenize('1101011'), [embedding.tokenize('0111111'), embedding.tokenize('1111110')])

# Create constant register values
constant_register_values = []
length = len(input_text)

constant_register_values.append('0' + '1' * (length - 1))  # tchaikovsky
constant_register_values.append('1' * (length - 1) + '0')  # anti_tchaikovsky
constant_register_values.append('0' * length)  # zeros
constant_register_values.append('1' * length)  # ones

constant_register_values.extend(user_defined_constant_register_values)

example_line = f"first_input = embedding.embed(embedding.tokenize('{input_text}'), ["

for register_value in constant_register_values:
    example_line += f"embedding.itokenize('{register_value}'), "

example_line = example_line[:-2] + "])\n"

f.write(example_line)

# Now, we actually create the program

func_templates = {
    "copy": "Copy(embedding, pos, <a>, <b>)",
    "copy_input": "ConvertToInternal(embedding, <a>)",
    "keep_": "Keep(<a>, <b>)",
    "rotate_": "Rotate(embedding, pos, tchaikovsky, anti_tchaikovsky, <a>, <b>, work_registers)",
    "rotate_with_limit_": "RotateWithLimit(embedding, pos, tchaikovsky, anti_tchaikovsky, <a>, <b>, <c>, work_registers)",
    "shiftr_": "Shift(embedding, pos, tchaikovsky, <a>, <b>, work_registers)",
    "shiftl_": "ShiftL(embedding, pos, anti_tchaikovsky, <a>, <b>, work_registers)",
    "xor": "XOR(embedding, pos, <a>, <b>, <c>, work_registers)",
    "and": "AND(embedding, <a>, <b>, <c>)",
    "not_": "NOT(embedding, pos, <a>, work_registers)",
    "print_": "Print(embedding, <a>)",
    "add": "Add(embedding, pos, anti_tchaikovsky, <a>, <b>, <c>, work_registers)"
}


def get_template(func_name, args):
    if func_name == "shift_":
        if int(args[1]) >= 0:
            func_name = "shiftr_"
        else:
            func_name = "shiftl_"

    template = func_templates.get(func_name)
    if template is None:
        raise CompilerError(f"Unknown function {func_name}")

    for i, arg in enumerate(args):
        template = template.replace(f"<{chr(97 + i)}>", arg)

    return template


f.write(f"pbar = tqdm(total={len(program_lines)}, leave=False)\n")

real_index = 0
for idx, line in enumerate(program_lines):
    f.write(f"pbar.update({idx})\n")
    # First, check if there's an assignment happening.
    if "=" in line:
        # In this case, split the line into the destination and the function
        dest, func = line.split("=")

        # Remove whitespace
        dest = dest.strip()
        func = func.strip()

        # Get the function name and arguments
        func_name = func.split("(")[0]
        args = func.split("(")[1].split(")")[0].split(",")

        # Remove whitespace from the arguments
        args = [arg.strip() for arg in args]
        # Get rid of empty arguments
        args = [arg for arg in args if arg != ""]
        # Add the destination to the arguments
        args.append(dest)

        # Write the actual function call
        template = get_template(func_name, args)

        f.write(f"op_{real_index} = {template}\n")
        real_index += 1
    elif len(line) > 2 and not line.strip().startswith("%"):
        # Otherwise, it's an in-place operation
        # Get the function name and arguments
        func_name = line.split("(")[0]
        args = line.split("(")[1].split(")")[0].split(",")

        # Remove whitespace from the arguments
        args = [arg.strip() for arg in args]
        # Get rid of empty arguments
        args = [arg for arg in args if arg != ""]

        if func_name == "del":
            # Special case for del
            ls = "["
            for arg in args:
                ls += arg + ", "
            ls = ls[:-2] + "]"
            f.write(f"op_{real_index} = Clear(embedding, {ls})\n")
        else:
            # Write the actual function call
            template = get_template(func_name, args)

            f.write(f"op_{real_index} = {template}\n")

        real_index += 1

f.write("pbar.close()\n")
f.write("print('ok we done')\n")

all_ops_string = ",\n    ".join([f"op_{i}" for i in range(real_index)])
f.write(f"""
all_ops = [
    {all_ops_string}
]
""")
f.write("""
def count_parameters(module):
    total_params = 0
    for _, param in module.state_dict().items():
        total_params += param.numel()

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, torch.nn.Module):
            total_params += count_parameters(attr)
        elif isinstance(attr, list):
            for item in attr:
                if isinstance(item, torch.nn.Module):
                    total_params += count_parameters(item)

    return total_params

s = 0

for i, module in enumerate(all_ops):
    total_params = count_parameters(module)
    print(f"op_{i}: Total parameters = {total_params}")
    s += total_params

print(f"Total parameters: {s}")
""")

f.write("""
x = op_0.forward(first_input.unsqueeze(0))[0]
# plot_tensor(x, embedding, 'op_0')
""")
for i in range(1, real_index):
    f.write(f"x = op_{i}.forward(x.unsqueeze(0))[0]\n")
    # f.write(f"plot_tensor(x, embedding, 'op_{i}')\n")

f.write("""
elapsed = time.time() - t
print(f"Elapsed time: {elapsed:.2f}s")
""")

f.close()
