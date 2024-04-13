# Various imports

import math

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from constants import INPUT_LENGTH

# A constant which determines how fine-grained our positional embedding is
POS_STEP = 1e-3


class Register(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.offset = None


class EmbeddedState(object):
    def __init__(self, tokens: list[str], registers: list[Register]):
        self.tokens = tokens
        self.token_map = {t: i for i, t in enumerate(tokens)}
        self.registers = registers
        self.register_map = {}
        self.register_size = 0

        if len(registers) == 0 or registers[0].name != 'pos':
            raise Exception("First register must be 'pos'")

        offset = len(tokens)
        for reg in registers:
            reg.offset = offset
            offset += reg.size
            self.register_size += reg.size
            self.register_map[reg.name] = reg

        self.dim = len(tokens) + self.register_size

    def tokenize(self, string: str):
        return F.one_hot(torch.tensor([self.token_map[c] for c in string]), num_classes=len(self.tokens)).float()

    def itokenize(self, string: str):
        return torch.tensor([self.token_map[c] for c in string]).float().unsqueeze(1)

    def embed(self, sequence, additional_constants):
        # We want to create additional space to store the registers
        extension_tensor = torch.zeros(*sequence.shape[:-1], self.register_size)

        # Encode position in the first extra embedding dimension
        for i in range(sequence.shape[0]):
            extension_tensor[i, 0] = math.sin(i * (2 * math.pi) * POS_STEP)
            extension_tensor[i, 1] = math.cos(i * (2 * math.pi) * POS_STEP)

        # Next columns of the extension tensor are the additional constants
        offset = 2
        for constant in additional_constants:
            extension_tensor[:, offset:offset + constant.shape[-1]] = constant
            offset += constant.shape[-1]

        sequence = torch.cat((sequence, extension_tensor), dim=-1)

        return sequence

    def predict(self, sequence):
        return self.tokens[torch.argmax(sequence[-1, :len(self.tokens)])]


class AttentionLayer(torch.nn.Module):
    def __init__(self, instruction):
        super(AttentionLayer, self).__init__()

        self.key = torch.nn.Parameter(instruction.key)
        self.value = torch.nn.Parameter(instruction.value)
        self.query = torch.nn.Parameter(instruction.query)

        self.mask = instruction.mask

        self.softmax = torch.nn.Softmax(2)

    def forward(self, seq):
        batch_size, seq_length, dim = seq.shape

        query = seq @ self.query
        key = seq @ self.key
        value = seq @ self.value

        causal_mask = (torch.triu(torch.ones(seq_length, seq_length), diagonal=1) * 0).to(seq.device)
        norm = np.sqrt(dim)

        kq = self.softmax(query @ key.transpose(-2, -1) / norm + causal_mask)

        s = (kq @ value) * self.mask

        return (seq + s)

    def reset(self):
        torch.nn.init.xavier_uniform_(self.key)
        torch.nn.init.xavier_uniform_(self.query)
        torch.nn.init.xavier_uniform_(self.value)


class GetRelativeToken(AttentionLayer):
    def __init__(self, embedding: EmbeddedState, pos_reg: Register, steps: int, out: Register):
        tpos_reg = embedding.register_map['pos']

        position_select = torch.zeros(embedding.dim, embedding.dim)
        position_select[tpos_reg.offset, tpos_reg.offset] = 1e10
        position_select[tpos_reg.offset + 1, tpos_reg.offset + 1] = 1e10

        i = -steps
        sin = math.sin(i * (2 * math.pi) * POS_STEP) * 1
        cos = math.cos(i * (2 * math.pi) * POS_STEP) * 1

        rotation = torch.zeros(embedding.dim, embedding.dim)
        rotation[pos_reg.offset, tpos_reg.offset] = cos
        rotation[pos_reg.offset + 1, tpos_reg.offset] = -sin
        rotation[pos_reg.offset, tpos_reg.offset + 1] = sin
        rotation[pos_reg.offset + 1, tpos_reg.offset + 1] = cos

        token_copy = torch.zeros(embedding.dim, embedding.dim)
        for i in range(len(embedding.tokens)):
            token_copy[i, i + out.offset] = 1.0

        self.query = rotation
        self.key = position_select
        self.value = token_copy

        self.mask = torch.zeros(embedding.dim)
        self.mask[out.offset:out.offset + out.size] = 1.0

        super(GetRelativeToken, self).__init__(self)


class MLPLayer(torch.nn.Module):
    def __init__(self, instruction, debug=False):
        super(MLPLayer, self).__init__()
        self.debug = debug

        self.first_weights = torch.nn.Parameter(instruction.first_weights)
        self.first_bias = torch.nn.Parameter(instruction.first_bias)
        self.second_weights = torch.nn.Parameter(instruction.second_weights)
        self.second_bias = torch.nn.Parameter(instruction.second_bias)

        self.gelu = torch.nn.ReLU()

        self.mask = instruction.mask

    def forward(self, seq):
        a = self.gelu(seq @ self.first_weights + self.first_bias)
        b = (a @ self.second_weights)
        x = b + self.second_bias
        return seq + (x * self.mask)

    def reset(self):
        torch.nn.init.xavier_uniform_(self.first_weights)
        torch.nn.init.zeros_(self.first_bias)
        torch.nn.init.xavier_uniform_(self.second_weights)
        torch.nn.init.zeros_(self.second_bias)


class ConvertToInternal(MLPLayer):
    def __init__(self, embedding: EmbeddedState, out: Register):
        self.first_weights = torch.zeros(embedding.dim, embedding.dim)
        self.first_bias = torch.zeros(embedding.dim)

        self.first_weights[1, out.offset] += 1

        self.second_weights = torch.eye(embedding.dim)
        self.second_bias = torch.zeros(embedding.dim)

        self.mask = torch.zeros(embedding.dim)
        for reg in [out]:
            self.mask[reg.offset:reg.offset + reg.size] = 1.0

        super(ConvertToInternal, self).__init__(self)


class GRLT2(AttentionLayer):
    """
    Copy the tokens from the given register to the output register, with an optional rotation by `steps`
    """

    def __init__(self, embedding: EmbeddedState, pos_reg: Register, steps: int, copy_from: Register, out: Register):
        tpos_reg = embedding.register_map['pos']

        position_select = torch.zeros(embedding.dim, embedding.dim)
        position_select[tpos_reg.offset, tpos_reg.offset] = 1e10
        position_select[tpos_reg.offset + 1, tpos_reg.offset + 1] = 1e10

        i = -steps
        sin = math.sin(i * (2 * math.pi) * POS_STEP) * 1
        cos = math.cos(i * (2 * math.pi) * POS_STEP) * 1

        rotation = torch.zeros(embedding.dim, embedding.dim)
        rotation[pos_reg.offset, tpos_reg.offset] = cos
        rotation[pos_reg.offset + 1, tpos_reg.offset] = -sin
        rotation[pos_reg.offset, tpos_reg.offset + 1] = sin
        rotation[pos_reg.offset + 1, tpos_reg.offset + 1] = cos

        token_copy = torch.zeros(embedding.dim, embedding.dim)
        token_copy[copy_from.offset, out.offset] = 1.0

        self.query = rotation
        self.key = position_select
        self.value = token_copy

        self.mask = torch.zeros(embedding.dim)
        self.mask[out.offset:out.offset + out.size] = 1.0

        super(GRLT2, self).__init__(self)


class AND(MLPLayer):
    def __init__(self, embedding: EmbeddedState, first_reg: Register, second_reg: Register, result_reg: Register):
        self.first_weights = torch.zeros(embedding.dim, embedding.dim)
        self.first_bias = torch.zeros(embedding.dim)

        self.first_weights[first_reg.offset, result_reg.offset] += 1
        self.first_weights[second_reg.offset, result_reg.offset] += 1
        self.first_bias[result_reg.offset:result_reg.offset + result_reg.size] = -1.0

        self.second_weights = torch.eye(embedding.dim)
        self.second_bias = torch.zeros(embedding.dim)

        self.mask = torch.zeros(embedding.dim)
        for reg in [result_reg]:
            self.mask[reg.offset:reg.offset + reg.size] = 1.0

        super(AND, self).__init__(self)


class Clear(MLPLayer):
    def __init__(self, embedding: EmbeddedState, registers: list[Register]):
        self.first_weights = torch.zeros(embedding.dim, embedding.dim)
        self.first_bias = torch.zeros(embedding.dim)

        for reg in registers:
            for i in range(reg.size):
                self.first_weights[reg.offset + i, reg.offset + i] = 100.0

        self.second_weights = torch.zeros(embedding.dim, embedding.dim)
        self.second_bias = torch.zeros(embedding.dim)
        for reg in registers:
            for i in range(reg.size):
                self.second_weights[reg.offset + i, reg.offset + i] = -0.01

        self.mask = torch.zeros(embedding.dim)
        for reg in registers:
            self.mask[reg.offset:reg.offset + reg.size] = 1.0

        super(Clear, self).__init__(self)


class Copy(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, pos_reg: Register, copy_from: Register, copy_to: Register):
        super(Copy, self).__init__()

        self.copy = GRLT2(embedding, pos_reg, 0, copy_from, copy_to)

    def forward(self, seq):
        return self.copy.forward(seq)


class Shift(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, pos: Register, tchaikovsky: Register, register_to_shift: Register,
                 amount: int, work_registers: list[Register]):
        super(Shift, self).__init__()

        self.embedding = embedding

        self.shiftpt1 = GRLT2(embedding, pos, -amount, register_to_shift, work_registers[0])
        self.clear = Clear(embedding, [register_to_shift])
        self.shifted_tchaikovsky = GRLT2(embedding, pos, -(amount - 1), tchaikovsky, work_registers[1])
        self.shiftpt2 = AND(embedding, work_registers[1], work_registers[0], register_to_shift)
        self.cleannup = Clear(embedding, work_registers)

    def forward(self, seq):
        x = self.shiftpt1.forward(seq)
        x = self.clear.forward(x)
        x = self.shifted_tchaikovsky.forward(x)
        x = self.shiftpt2.forward(x)
        x = self.cleannup.forward(x)
        return x


class ShiftL(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, pos: Register, anti_tchaikovsky: Register, register_to_shift: Register,
                 amount: int, work_registers: list[Register]):
        super(ShiftL, self).__init__()

        self.embedding = embedding

        self.shiftpt1 = GRLT2(embedding, pos, amount, register_to_shift, work_registers[0])
        self.clear = Clear(embedding, [register_to_shift])
        self.shifted_antitchaikovsky = GRLT2(embedding, pos, amount - 1, anti_tchaikovsky, work_registers[1])
        self.shiftpt2 = AND(embedding, work_registers[1], work_registers[0], register_to_shift)
        self.cleannup = Clear(embedding, work_registers)

    def forward(self, seq):
        x = self.shiftpt1.forward(seq)
        x = self.clear.forward(x)
        x = self.shifted_antitchaikovsky.forward(x)
        x = self.shiftpt2.forward(x)
        x = self.cleannup.forward(x)
        return x


class NOT_To(MLPLayer):
    def __init__(self, embedding: EmbeddedState, from_reg: Register, result_reg: Register):
        self.first_weights = torch.zeros(embedding.dim, embedding.dim)
        self.first_bias = torch.zeros(embedding.dim)

        self.first_weights[from_reg.offset, result_reg.offset] = -1
        self.first_bias[result_reg.offset:result_reg.offset + result_reg.size] = 1.0

        self.second_weights = torch.eye(embedding.dim)
        self.second_bias = torch.zeros(embedding.dim)

        self.mask = torch.zeros(embedding.dim)
        for reg in [result_reg]:
            self.mask[reg.offset:reg.offset + reg.size] = 1.0

        super(NOT_To, self).__init__(self)


class NOT(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, pos: Register, register: Register, work_registers: list[Register]):
        super(NOT, self).__init__()

        self.not_to = NOT_To(embedding, register, work_registers[0])
        self.clear = Clear(embedding, [register])
        self.copy = Copy(embedding, pos, work_registers[0], register)
        self.clear2 = Clear(embedding, work_registers)

    def forward(self, seq):
        x = self.not_to.forward(seq)
        x = self.clear.forward(x)
        x = self.copy.forward(x)
        x = self.clear2.forward(x)
        return x


class NOR(MLPLayer):
    def __init__(self, embedding: EmbeddedState, first_reg: Register, second_reg: Register, result_reg: Register):
        self.first_weights = torch.zeros(embedding.dim, embedding.dim)
        self.first_bias = torch.zeros(embedding.dim)

        self.first_weights[first_reg.offset, result_reg.offset] += -1
        self.first_weights[second_reg.offset, result_reg.offset] += -1
        self.first_bias[result_reg.offset:result_reg.offset + result_reg.size] = 1.0

        self.second_weights = torch.eye(embedding.dim)
        self.second_bias = torch.zeros(embedding.dim)

        self.mask = torch.zeros(embedding.dim)
        for reg in [result_reg]:
            self.mask[reg.offset:reg.offset + reg.size] = 1.0

        super(NOR, self).__init__(self)


class OR(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, pos: Register, first_reg: Register, second_reg: Register,
                 result_reg: Register, work_registers: list[Register]):
        super(OR, self).__init__()

        self.part1 = NOR(embedding, first_reg, second_reg, result_reg)
        self.part2 = NOT(embedding, pos, result_reg, work_registers)
        self.cleanup = Clear(embedding, work_registers)

    def forward(self, seq):
        x = self.part1.forward(seq)
        x = self.part2.forward(x)
        x = self.cleanup.forward(x)
        return x


class XOR(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, pos: Register, first_reg: Register, second_reg: Register,
                 result_reg: Register, work_registers: list[Register]):
        super(XOR, self).__init__()

        # A OR B
        self.part1 = OR(embedding, pos, first_reg, second_reg, work_registers[0], work_registers[1:])

        # A AND B
        self.part2 = AND(embedding, first_reg, second_reg, work_registers[1])

        # NOT (A AND B)
        self.part3 = NOT(embedding, pos, work_registers[1], work_registers[2:])

        # (A OR B) AND NOT (A AND B)
        self.part4 = AND(embedding, work_registers[0], work_registers[1], result_reg)

        # Clear the work registers
        self.part5 = Clear(embedding, work_registers)

    def forward(self, seq):
        x = self.part1.forward(seq)
        x = self.part2.forward(x)
        x = self.part3.forward(x)
        x = self.part4.forward(x)
        x = self.part5.forward(x)
        return x


class Rotate(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, pos: Register, tchaikovsky: Register, anti_tchaikovsky: Register,
                 register_to_rotate: Register, amount: int, work_registers: list[Register]):
        super(Rotate, self).__init__()

        self.embedding = embedding

        # First, we need to copy the register to two work registers
        # Thus, work registers 0 and 1 are currently in use
        self.copies = [Copy(embedding, pos, register_to_rotate, work_registers[i]) for i in range(2)]

        # Next, we shift the first work register to the right
        self.shift_right = Shift(embedding, pos, tchaikovsky, work_registers[0], amount, work_registers[2:])

        # Then, we shift the second work register to the left INPUT_LENGTH - 1 times
        self.left_shifts = ShiftL(embedding, pos, anti_tchaikovsky, work_registers[1], (INPUT_LENGTH - amount),
                                  work_registers[2:])

        # Now, we clear the original register
        self.clear = Clear(embedding, [register_to_rotate])

        # And finally, we OR work registers 0 and 1 to get the final result
        self.or_result = OR(embedding, pos, work_registers[0], work_registers[1], register_to_rotate,
                            work_registers[2:])

        # Oh, and clear the work registers
        self.clear_work = Clear(embedding, work_registers)

    def forward(self, seq):
        for copy in self.copies:
            seq = copy.forward(seq)
        seq = self.shift_right.forward(seq)
        seq = self.left_shifts.forward(seq)
        seq = self.clear.forward(seq)
        seq = self.or_result.forward(seq)
        seq = self.clear_work.forward(seq)
        return seq


class Add(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, pos: Register, anti_tchaikovsky: Register, a: Register, b: Register,
                 result: Register, work_registers: list[Register]):
        super(Add, self).__init__()

        self.embedding = embedding

        # work_registers[0] is `sum_`
        # work_registers[1] is `carry`
        self.first_sum = XOR(embedding, pos, a, b, work_registers[0], work_registers[2:])
        self.first_carry = AND(embedding, a, b, work_registers[1])

        self.next_operations = []

        for _ in range(32):
            # Copy `carry` to work_registers[2]
            copy_of_carry = Copy(embedding, pos, work_registers[1], work_registers[2])
            # Shift this copy of `carry` to the left. Now `work_registers[2]` contains `shifted_carry`
            shifted_carry = ShiftL(embedding, pos, anti_tchaikovsky, work_registers[2], 1, work_registers[3:])
            # XOR `sum_` with `shifted_carry`. Now `work_registers[3]` contains `new_sum`
            new_sum = XOR(embedding, pos, work_registers[0], work_registers[2], work_registers[3], work_registers[4:])
            # Clear `carry`
            clear_carry = Clear(embedding, [work_registers[1]])
            # AND `sum_` with `shifted_carry`. Now `work_registers[1]` contains `carry` again
            carry = AND(embedding, work_registers[0], work_registers[2], work_registers[1])
            # Clear `sum`
            clear_sum = Clear(embedding, [work_registers[0]])
            # Copy `new_sum` to `sum_`
            sum = Copy(embedding, pos, work_registers[3], work_registers[0])
            # Clear the work registers
            clear_work = Clear(embedding, work_registers[2:])
            self.next_operations.append(
                (copy_of_carry, shifted_carry, new_sum, clear_carry, carry, clear_sum, sum, clear_work))

        self.copy_to_result = Copy(embedding, pos, work_registers[0], result)

        self.clear = Clear(embedding, work_registers)

    def forward(self, seq):
        x = self.first_sum.forward(seq)
        x = self.first_carry.forward(x)

        for copy_of_carry, shifted_carry, new_sum, clear_carry, carry, clear_sum, sum, clear_work in self.next_operations:
            x = copy_of_carry.forward(x)
            x = shifted_carry.forward(x)
            x = new_sum.forward(x)
            x = clear_carry.forward(x)
            x = carry.forward(x)
            x = clear_sum.forward(x)
            x = sum.forward(x)
            x = clear_work.forward(x)

        x = self.copy_to_result.forward(x)
        x = self.clear.forward(x)

        return x


class RotateWithLimit(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, pos: Register, tchaikovsky: Register, anti_tchaikovsky: Register,
                 register_to_rotate: Register, amount: int, limit: int, work_registers: list[Register]):
        super(RotateWithLimit, self).__init__()

        self.embedding = embedding

        # First, we need to copy the register to two work registers
        # Thus, work registers 0 and 1 are currently in use
        self.copies = [Copy(embedding, pos, register_to_rotate, work_registers[i]) for i in range(2)]

        # Next, we shift the first work register to the right
        self.shift_right = Shift(embedding, pos, tchaikovsky, work_registers[0], amount, work_registers[2:])

        # Then, we shift the second work register to the left `limit - amount` times
        self.left_shifts = ShiftL(embedding, pos, anti_tchaikovsky, work_registers[1], limit - amount,
                                  work_registers[2:])

        # Now we clear the rest of the `head` register to zeros to allow it to or in
        self.clear_shift_1 = Shift(embedding, pos, tchaikovsky, work_registers[1], INPUT_LENGTH - amount,
                                   work_registers[2:])
        self.clear_shift_2 = ShiftL(embedding, pos, anti_tchaikovsky, work_registers[1],
                                    INPUT_LENGTH - amount, work_registers[2:])

        # Now, we clear the original register
        self.clear = Clear(embedding, [register_to_rotate])

        # And finally, we OR work registers 0 and 1 to get the final result
        self.or_result = OR(embedding, pos, work_registers[0], work_registers[1], register_to_rotate,
                            work_registers[2:])

        # Oh, and clear the work registers
        self.clear_work = Clear(embedding, work_registers)

    def forward(self, seq):
        for copy in self.copies:
            seq = copy.forward(seq)
        seq = self.shift_right.forward(seq)
        seq = self.left_shifts.forward(seq)
        seq = self.clear_shift_1.forward(seq)
        seq = self.clear_shift_2.forward(seq)
        seq = self.clear.forward(seq)
        seq = self.or_result.forward(seq)
        seq = self.clear_work.forward(seq)
        return seq


class Print(torch.nn.Module):
    def __init__(self, embedding: EmbeddedState, register: Register):
        super(Print, self).__init__()

        self.embedding = embedding
        self.register = register

    def forward(self, seq):
        print(''.join(
            str(c) for c in [int(q) for q in seq[0, :, self.register.offset:self.register.offset + self.register.size]
            .detach().flatten()]))

        return seq
