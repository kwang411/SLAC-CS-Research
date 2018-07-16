def answer(start, length):
    start_nums = [start + i*length for i in range(length)]
    temp = 0
    for i in range(length):
        temp_length = length - i
        for j in range(temp_length):
            temp ^= (start_nums[i] + j)
    return temp

# print(answer(0, 3))
# print(answer(17, 4))

def answer(n):
    memo = {}
    return (recurse(n, 0, memo) - 1) # subtract 1 to remove case where only one step
    
def recurse(remainder, recent, memo):
    if remainder == 0:
        return 1
    if remainder <= recent:
        return 0
    if (remainder, recent) in memo:
        return memo[(remainder, recent)]
    result = sum(recurse(remainder-i, i, memo) for i in range(recent+1, remainder+1))
    memo[(remainder, recent)] = result
    return result

# for i in range(3,15):
# 	print(answer(i))
import fractions as f
from fractions import Fraction
epsilon = 10**-12

def answer(m):
	terminal = []
	for i in range(len(m)):
		if isZero(m[i]):
			terminal.append(i)
			m[i][i] = 1
			continue
		m[i] = normalize(m[i])
	m = transpose(m)
	
	initial = [0 for i in range(len(m))]
	initial[0] = 1

	while(True):
		temp = multiply(m, initial)
		if distance(temp, initial) < epsilon:
			initial = temp
			break
		initial = temp

	fractions = []
	for i in terminal:
		fractions.append(Fraction(initial[i]).limit_denominator())
	common_denom = lcm_list([fractions[i].denominator for i in range(len(fractions))])

	result = [int(fractions[i].numerator*common_denom/fractions[i].denominator) for i in range(len(fractions))]
	result.append(int(common_denom))
	return result

def isZero(l):
	for i in range(len(l)):
		if l[i] != 0:
			return False
	return True

def normalize(l):
	total = sum(l[i] for i in range(len(l)))
	return [l[i]/(1.0*total) for i in range(len(l))]

def transpose(m):
	return [[row[i] for row in m] for i in range(len(m[0]))]

def multiply(m, l):
	return [sum(m[i][j]*l[j] for j in range(len(l))) for i in range(len(l))]

def lcm(a, b):
    return a*b/(1.0*f.gcd(a, b))

def lcm_list(l):
    return reduce(lcm, l)

def distance(k, l):
	return sum((k[i] - l[i])**2 for i in range(len(k)))**0.5

print(answer([[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
print(answer([[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))
