from enum import IntEnum
import numpy as np
import math
from math import factorial


class ParseStatus(IntEnum):
    CORRECT = 1
    BAD_FORMAT = 2
    PARTIAL_CORRECT = 3
    INCORRECT = 4


def parse_atomic_eq(atomic_eq):
    tks = atomic_eq.split('=')
    if len(tks) != 2:
        return ParseStatus.BAD_FORMAT
    for op in ['+', '*', '/', '-', '^', '!']:
        if tks[0][0] == '-':
            xs = tks[0][1:].split(op, 1)
            xs[0] = '-' + xs[0]
        elif tks[0][0] == '+':
            xs = tks[0][1:].split(op, 1)
            xs[0] = '+' + xs[0]
        else:
            xs = tks[0].split(op, 1)
        if len(xs) == 2:
            if xs[0][0] == '+' or tks[1][0] == '+':
                return ParseStatus.BAD_FORMAT
            if op == '!':
                if xs[-1] != '' or int(xs[0]) < 0:
                    return ParseStatus.BAD_FORMAT
                else:
                    # (a, None, !, result)
                    return int(xs[0]), None, op, int(tks[1])
            if xs[1][0] == '+':
                return ParseStatus.BAD_FORMAT
            # (a, b, op, result)
            return int(xs[0]), int(xs[1]), op, int(tks[1])

    # not valid op
    return ParseStatus.BAD_FORMAT


def parse_seq(seq):
    # remove any unknown char
    seq = seq.replace('U', "")

    # '{prompt}S{response}
    items = seq.strip().split('S')
    if len(items) != 2:
        return ParseStatus.BAD_FORMAT
    prompt = items[0]
    response = items[1]

    # num_1,...,num_n:target
    prompt_items = prompt.split(':')
    if len(prompt_items) != 2:
        return ParseStatus.BAD_FORMAT

    target = int(prompt_items[1])

    nums = prompt_items[0]
    nums = [int(x) for x in nums.split(',')]

    # e.g., 'a*b=x,c-d=y,y!=z,z^x=t'
    eqs = response.split(',')
    for atomic_eq in eqs:
        ret = parse_atomic_eq(atomic_eq)
        if not isinstance(ret, tuple):
            return ParseStatus.BAD_FORMAT
        a, b, op, result = ret
        if a not in nums:
            return ParseStatus.INCORRECT
        nums.remove(a)
        if op == '!':
            if b != None:
                return ParseStatus.BAD_FORMAT
            if a < 0:
                # cannot apply factorial to negative integer
                return ParseStatus.INCORRECT
            r = factorial(a)
            if r != result:
                return ParseStatus.INCORRECT
        else:
            if b not in nums:
                return ParseStatus.INCORRECT
            nums.remove(b)
            if op == '+':
                r = a + b
                if r != result:
                    return ParseStatus.INCORRECT
            elif op == '-':
                r = a - b
                if r != result:
                    return ParseStatus.INCORRECT
            elif op == '*':
                r = a * b
                if r != result:
                    return ParseStatus.INCORRECT
            elif op == '/':
                if b == 0:
                    return ParseStatus.INCORRECT
                # we allow floor division
                # if a % b != 0:
                #     return 0.0
                r = a // b
                if r != result:
                    return ParseStatus.INCORRECT
            elif op == '^':
                r = math.floor(a ** b)
                if r != result:
                    return ParseStatus.INCORRECT
            else:
                return ParseStatus.BAD_FORMAT
        nums.append(result)

    # must use up all nums and all intermediate results
    if len(nums) != 1:
        return ParseStatus.INCORRECT

    if nums[0] != target:
        # the process is correct, but the final result is incorrect
        return ParseStatus.PARTIAL_CORRECT

    # all tests passed, a correct response
    return ParseStatus.CORRECT


def reward(seq, partial_correct_reward=0.1, bad_format_reward=0.0):
    # loose EOS treatment: the sub string for the first occurrence of EOS token
    idx = seq.find('E')
    if idx > -1:
        seq = seq[0:idx]
    # remove any UNKNOWN token, PAD token
    seq = seq.replace('U', "")
    seq = seq.replace('P', "")
    try:
        ret = parse_seq(seq)
    except IndexError as ex:
        # print(f'Index exception: {ex}')
        return bad_format_reward
    except ValueError as ex:
        # print(f'Value exception: {ex}')
        return bad_format_reward
    except Exception as ex:
        print(f'Unknown exception: {ex} \nseq: {seq}')
        return bad_format_reward

    if ret == ParseStatus.INCORRECT:
        return 0.0
    if ret == ParseStatus.PARTIAL_CORRECT:
        return partial_correct_reward
    if ret == ParseStatus.BAD_FORMAT:
        return bad_format_reward
    if ret == ParseStatus.CORRECT:
        return 1.0


if __name__ == '__main__':
    def float_eq(a, b):
        return True if b - 1e-8 < a < b + 1e-8 else False

    ### special token test
    # correct: padding numbers after eos
    seq = 'PPPPP5,3,2,10:70S5!=120,3+2=5,5*10=50,120-50=70UE313213'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # incorrect: parentheses is illegal
    seq = 'PPPPP5,3,2,10:70S5!=120,3+(2)=5,5*10=50,120-50=70UE313213'
    rwd = reward(seq)
    assert float_eq(rwd, 0.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # correct, space bettewen numbers
    seq = '5,5,9,11:189S11+5= 16 , 16+ 5=21,21*9=189U'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # incorrect
    seq = '5,5,9,11:189S11+5=16,XYZ16+5=21,21*9=189U'
    rwd = reward(seq)
    assert float_eq(rwd, 0.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # correct: unknow/pading tokens will be ignored 
    seq = 'PPPU2,3,6,9,9,11:8S9+11U=20,U2*9U=18U,2U0/18=1,1-3P=-2,6--2=8PE'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # correct: space will be ignored
    seq = 'PPPPP2,3,3:24S2^3 =8,3*8= 24'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # incorrect: space cannot in the middle of a number
    seq = 'PPPPP2,3,3:24S2^3 =8,3*8= 2 4'
    rwd = reward(seq)
    assert float_eq(rwd, 0.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # incorrect
    seq = '2,3,6,9,9,11:8S9+11=20,2*9=18,20/18=1,1-3=-2,6-- 2=8'
    rwd = reward(seq)
    assert float_eq(rwd, 0.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # correct: ignored tokens after first E
    seq = 'PPPPPP2,3,6,9,9,11:8S9+11=20,2*9=18,20/18=1,1-3=-2,6--2=8E8E8E8E8'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    ### negetive/positive sign test
    # correct: negetive zero will be treated as zero
    seq = 'PPPPP2,3,3:-3S2/3=-0,-0-3=-3'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # incorrect seq2: positive can only be an operator
    seq1 = 'PPPPP2,3,10,11:-60S2-11=-9,-9+3=-6,10*-6=-60'
    seq2 = 'PPPPP2,3,10,11:-60S2-11=-9,-9+3=-6,+10*-6=-60'
    rwd1 = reward(seq1)
    rwd2 = reward(seq2)
    assert float_eq(rwd1, 1.0)
    assert float_eq(rwd2, 0.0)
    print(f'rwd1: {rwd1}, seq: {seq1}')
    print(f'rwd2: {rwd2}, seq: {seq2}')

    ### functional test
    # correct: not use up all nums and all intermediate results
    seq = '3,8,4,12:24S3*8=24,4*12=48'
    rwd = reward(seq, partial_correct_reward=0.0, bad_format_reward=0.0)
    assert float_eq(rwd, 0.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # incorrect final result
    seq = '5,5,9,11:181S11+5=16,16+5=21,21*9=181U'
    rwd = reward(seq)
    assert float_eq(rwd, 0.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # correct +-*/, n=5
    seq = '2,5,6,7,11:-42S5*7=35,2+11=13,6-13=-7,-7-35=-42'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # correct, +-*/^, n=6
    seq = '1,5,5,5,9,11:189S1^5=1,11+5=16,16+5=21,21*9=189,189*1=189U'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # correct +-*/, n=7
    seq = '5,5,11,11,11,12,12:68S5-12=-7,-7+11=4,11-11=0,0-12=-12,5--12=17,17*4=68'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # incorrect final result
    seq = '2,5,6,7,11:-42S5*7=35,2+11=13,6-13=-7,-7-38=-42'
    rwd = reward(seq)
    assert float_eq(rwd, 0.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # partial correct, the final result is not the target
    seq = '1,3,7,9,10,10:24S3-1=2,9-10=-1,7--1=8,2+10=12,8-12=-4'
    rwd = reward(seq, partial_correct_reward=0.88)
    assert float_eq(rwd, 0.88)
    print(f'rwd: {rwd}, seq: {seq}')

    # incorrect, invalid nums
    seq = '1,3,7,9,10,10:24S3-1=2,9-13=2,7--1=8,2+10=12,8-12=-4'
    rwd = reward(seq)
    assert float_eq(rwd, 0.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # factorial
    seq = 'PPPPP3,4,5:50S3!=6,6+4=10,10*5=50'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # multiple factorial, which is correct
    seq = 'PPPPP3,4,5:729S3!=6,6!=720,720+4=724,724+5=729'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # pow, correct
    seq = 'PPPPP2,3,3:24S2^3=8,3*8=24'
    rwd = reward(seq)
    assert float_eq(rwd, 1.0)
    print(f'rwd: {rwd}, seq: {seq}')

    # pow, not use up all nums and intermediate results
    seq = 'PPPPP2,3,3,5:24S2^3=8,3*8=24'
    rwd = reward(seq)
    assert float_eq(rwd, 0.0)
    print(f'rwd: {rwd}, seq: {seq}')