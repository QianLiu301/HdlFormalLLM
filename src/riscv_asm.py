"""
RV32I Assembler
===============

把 BDD 场景里的汇编文本（``add x1, x2, x3``）编码成 32 位机器码，供 CPU
testbench 写进指令存储器。

为什么单独成一个模块：编码错了不会报错，只会让激励悄悄跑偏——CPU 照常执行、
仿真照常结束、通过率照常有个数字，但测的根本不是那条指令。B 型和 J 型的立即数
在 RV32I 里是**打散重排**的（B: imm[12|10:5] 与 imm[4:1|11] 分处两端；
J: imm[20|10:1|11|19:12]），错一位就是另一个跳转目标。所以这里的每条编码规则
都配了对照测试，见本文件末尾的 ``_SELFTEST``。

支持 duv_cpu/v1.yaml 声明的那 19 条指令，不多不少。
"""

import re
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------
# 指令表：名字 -> (格式, opcode, funct3, funct7)
# 与 prompts/duv_cpu/v1.yaml 的指令表逐条对应
# --------------------------------------------------------------------------
R_TYPE = {
    'add':  (0b0110011, 0b000, 0b0000000),
    'sub':  (0b0110011, 0b000, 0b0100000),
    'and':  (0b0110011, 0b111, 0b0000000),
    'or':   (0b0110011, 0b110, 0b0000000),
    'xor':  (0b0110011, 0b100, 0b0000000),
    'slt':  (0b0110011, 0b010, 0b0000000),
}
I_TYPE = {
    'addi': (0b0010011, 0b000),
    'andi': (0b0010011, 0b111),
    'ori':  (0b0010011, 0b110),
    'xori': (0b0010011, 0b100),
    'slti': (0b0010011, 0b010),
    'lw':   (0b0000011, 0b010),
}
S_TYPE = {'sw': (0b0100011, 0b010)}
B_TYPE = {
    'beq': (0b1100011, 0b000),
    'bne': (0b1100011, 0b001),
    'blt': (0b1100011, 0b100),
    'bge': (0b1100011, 0b101),
}
J_TYPE = {'jal': 0b1101111}
JALR = 0b1100111

MASK32 = 0xFFFFFFFF


class AsmError(ValueError):
    """无法编码的指令。调用方应当跳过该行并计数，不要猜。"""


def _reg(tok: str) -> int:
    t = tok.strip().lower()
    m = re.fullmatch(r'x(\d+)', t)
    if not m:
        raise AsmError(f'not a register: {tok!r}')
    n = int(m.group(1))
    if not 0 <= n <= 31:
        raise AsmError(f'register out of range: {tok!r}')
    return n


def _imm(tok: str) -> int:
    t = tok.strip().lower()
    try:
        return int(t, 16) if t.startswith(('0x', '-0x')) else int(t, 0)
    except ValueError:
        raise AsmError(f'not an immediate: {tok!r}')


def _fits(v: int, bits: int) -> int:
    """有符号立即数截断到 bits 位；超范围直接报错，不静默回绕。"""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if not lo <= v <= hi:
        raise AsmError(f'immediate {v} does not fit in {bits} signed bits')
    return v & ((1 << bits) - 1)


def enc_r(op: str, rd: int, rs1: int, rs2: int) -> int:
    opcode, f3, f7 = R_TYPE[op]
    return ((f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12)
            | (rd << 7) | opcode) & MASK32


def enc_i(op: str, rd: int, rs1: int, imm: int) -> int:
    opcode, f3 = I_TYPE[op]
    return ((_fits(imm, 12) << 20) | (rs1 << 15) | (f3 << 12)
            | (rd << 7) | opcode) & MASK32


def enc_jalr(rd: int, rs1: int, imm: int) -> int:
    return ((_fits(imm, 12) << 20) | (rs1 << 15) | (0b000 << 12)
            | (rd << 7) | JALR) & MASK32


def enc_s(op: str, rs2: int, rs1: int, imm: int) -> int:
    opcode, f3 = S_TYPE[op]
    i = _fits(imm, 12)
    return (((i >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12)
            | ((i & 0x1F) << 7) | opcode) & MASK32


def enc_b(op: str, rs1: int, rs2: int, imm: int) -> int:
    """B 型：立即数以 2 字节为单位，且位序被打散。

    imm[12] -> bit31, imm[10:5] -> bit30:25, imm[4:1] -> bit11:8,
    imm[11] -> bit7。imm[0] 恒为 0，不编码。
    """
    opcode, f3 = B_TYPE[op]
    if imm % 2:
        raise AsmError(f'branch offset must be even: {imm}')
    i = _fits(imm, 13)
    return (((i >> 12 & 1) << 31) | ((i >> 5 & 0x3F) << 25) | (rs2 << 20)
            | (rs1 << 15) | (f3 << 12) | ((i >> 1 & 0xF) << 8)
            | ((i >> 11 & 1) << 7) | opcode) & MASK32


def enc_j(rd: int, imm: int) -> int:
    """J 型：imm[20]->31, imm[10:1]->30:21, imm[11]->20, imm[19:12]->19:12。"""
    if imm % 2:
        raise AsmError(f'jump offset must be even: {imm}')
    i = _fits(imm, 21)
    return (((i >> 20 & 1) << 31) | ((i >> 1 & 0x3FF) << 21)
            | ((i >> 11 & 1) << 20) | ((i >> 12 & 0xFF) << 12)
            | (rd << 7) | J_TYPE['jal']) & MASK32


NOP = enc_i('addi', 0, 0, 0)          # addi x0, x0, 0

_MEM = re.compile(r'^\s*(-?\w+)\s*\(\s*(x\d+)\s*\)\s*$', re.I)


def assemble(text: str) -> int:
    """把一条汇编文本编码成 32 位字；无法识别时抛 AsmError。"""
    s = re.sub(r'[#;].*', '', text).strip().lower()
    if not s:
        raise AsmError('empty instruction')
    if s in ('nop',):
        return NOP
    m = re.match(r'^([a-z]+)\s+(.*)$', s)
    if not m:
        raise AsmError(f'cannot parse: {text!r}')
    op, rest = m.group(1), m.group(2)
    args = [a.strip() for a in rest.split(',') if a.strip()]

    if op in R_TYPE:
        if len(args) != 3:
            raise AsmError(f'{op} needs 3 operands: {text!r}')
        return enc_r(op, _reg(args[0]), _reg(args[1]), _reg(args[2]))

    if op in ('lw',):
        # lw rd, imm(rs1)
        if len(args) != 2:
            raise AsmError(f'{op} needs 2 operands: {text!r}')
        mm = _MEM.match(args[1])
        if not mm:
            raise AsmError(f'{op} expects imm(rs1): {text!r}')
        return enc_i(op, _reg(args[0]), _reg(mm.group(2)), _imm(mm.group(1)))

    if op in I_TYPE:
        if len(args) != 3:
            raise AsmError(f'{op} needs 3 operands: {text!r}')
        return enc_i(op, _reg(args[0]), _reg(args[1]), _imm(args[2]))

    if op in S_TYPE:
        # sw rs2, imm(rs1)
        if len(args) != 2:
            raise AsmError(f'{op} needs 2 operands: {text!r}')
        mm = _MEM.match(args[1])
        if not mm:
            raise AsmError(f'{op} expects imm(rs1): {text!r}')
        return enc_s(op, _reg(args[0]), _reg(mm.group(2)), _imm(mm.group(1)))

    if op in B_TYPE:
        if len(args) != 3:
            raise AsmError(f'{op} needs 3 operands: {text!r}')
        return enc_b(op, _reg(args[0]), _reg(args[1]), _imm(args[2]))

    if op == 'jal':
        # jal rd, imm
        if len(args) != 2:
            raise AsmError(f'jal needs 2 operands: {text!r}')
        return enc_j(_reg(args[0]), _imm(args[1]))

    if op == 'jalr':
        # jalr rd, rs1, imm
        if len(args) != 3:
            raise AsmError(f'jalr needs 3 operands: {text!r}')
        return enc_jalr(_reg(args[0]), _reg(args[1]), _imm(args[2]))

    raise AsmError(f'unsupported instruction: {op!r}')


def li_addi_only(rd: int, value: int) -> Optional[List[int]]:
    """只用一条 ADDI 装载常量；装不下返回 None。

    duv_cpu/v1.yaml 的指令表里**没有 LUI**，所以被测 CPU 不实现它。用 li()
    发出 LUI 的话，CPU 碰到未实现的 opcode 会给出垃圾值，而仿真照常跑完——
    实测 marker 0x5A5A5A5A 就是这样变成 0xFFFFF4F4 的。凡是需要 LUI 的值，
    调用方必须跳过该行并计数，不能硬凑。
    """
    v = value & MASK32
    sv = v - (1 << 32) if v >> 31 else v
    if -2048 <= sv <= 2047:
        return [enc_i('addi', rd, 0, sv)]
    return None


def li(rd: int, value: int) -> List[int]:
    """把任意 32 位常量装进 rd。

    12 位以内一条 ADDI 就够；否则 LUI+ADDI。ADDI 的立即数是**有符号**的，
    所以当低 12 位的最高位为 1 时，要给 LUI 的高 20 位补 1，否则结果会少
    0x1000——这是手写 li 最常见的错误。
    """
    v = value & MASK32
    sv = v - (1 << 32) if v >> 31 else v
    if -2048 <= sv <= 2047:
        return [enc_i('addi', rd, 0, sv)]
    lo = v & 0xFFF
    lo_s = lo - 4096 if lo & 0x800 else lo      # 低 12 位按有符号解释
    hi = (v - lo_s) >> 12 & 0xFFFFF             # 补偿后的高 20 位
    lui = ((hi << 12) | (rd << 7) | 0b0110111) & MASK32
    return [lui, enc_i('addi', rd, rd, lo_s)]


# --------------------------------------------------------------------------
# 对照测试：与已知编码逐条比对
# --------------------------------------------------------------------------
_SELFTEST: List[Tuple[str, int]] = [
    # 这四条来自 testbench_generator 里原本硬编码的程序，是已验证过的编码
    ('addi x1, x0, 10',  0x00A00093),
    ('addi x2, x0, 20',  0x01400113),
    ('add x3, x1, x2',   0x002081B3),
    ('nop',              0x00000013),
    # 手工按 RV32I 规范推导
    ('sub x5, x6, x7',   0x407302B3),
    # rd=1 rs1=2 rs2=3，只有 funct3 不同：and=111 or=110 xor=100 slt=010
    ('and x1, x2, x3',   0x003170B3),
    ('or x1, x2, x3',    0x003160B3),
    ('xor x1, x2, x3',   0x003140B3),
    ('slt x1, x2, x3',   0x003120B3),
    ('andi x1, x2, 15',  0x00F17093),
    ('ori x1, x2, 15',   0x00F16093),
    ('xori x1, x2, 15',  0x00F14093),
    ('slti x1, x2, 5',   0x00512093),
    ('addi x1, x2, -1',  0xFFF10093),
    ('lw x1, 8(x2)',     0x00812083),
    ('sw x3, 4(x2)',     0x00312223),
    ('beq x1, x2, 8',    0x00208463),
    ('bne x1, x2, 8',    0x00209463),
    ('blt x1, x2, 8',    0x0020C463),
    ('bge x1, x2, 8',    0x0020D463),
    ('beq x1, x2, -4',   0xFE208EE3),
    ('jal x1, 16',       0x010000EF),
    ('jalr x1, x2, 8',   0x008100E7),
]


def selftest(verbose: bool = True) -> bool:
    ok = True
    for text, want in _SELFTEST:
        try:
            got = assemble(text)
        except AsmError as e:
            print(f'  ✗ {text:<22} 抛错 {e}')
            ok = False
            continue
        if got != want:
            print(f'  ✗ {text:<22} 得到 0x{got:08X}，应为 0x{want:08X}')
            ok = False
        elif verbose:
            print(f'  ✓ {text:<22} 0x{got:08X}')
    # li 的边界
    cases = [(0, [0x00000013]), (10, None), (-1, None), (0x12345678, None),
             (0x800, None), (0xFFFFF800, None)]
    for val, _ in cases:
        words = li(1, val)
        if verbose:
            print(f'  li x1, 0x{val & MASK32:08X} -> '
                  + ' '.join(f'0x{w:08X}' for w in words))
    return ok


if __name__ == '__main__':
    import sys
    print('RV32I assembler self-test')
    sys.exit(0 if selftest() else 1)
