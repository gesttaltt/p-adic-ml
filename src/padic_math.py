
def mod_inverse(a, m):
    """Compute the modular inverse of a modulo m using the Extended Euclidean Algorithm."""
    a = a % m
    t, newt = 0, 1
    r, newr = m, a
    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr
    if r > 1:
        raise ValueError(f"{a} is not invertible modulo {m}")
    if t < 0:
        t = t + m
    return t

def rational_to_padic(r, s, p, N):
    """
    Generate the first N p-adic digits of the rational number r/s.
    s must be coprime to p.
    Returns a list of N integers in {0, ..., p-1}.
    """
    if s % p == 0:
        raise ValueError(f"Denominator {s} is not coprime to p={p}")
    
    digits = []
    curr_r = r
    curr_s = s
    
    # We want to solve curr_r / curr_s = a_0 + a_1 * p + ...
    # At each step:
    # a_i = (curr_r * curr_s^-1) % p
    # curr_r_next / curr_s = (curr_r / curr_s - a_i) / p => curr_r_next = (curr_r - a_i * curr_s) / p
    # Note that curr_s remains constant!
    s_inv = mod_inverse(curr_s, p)
    for _ in range(N):
        a = (curr_r * s_inv) % p
        digits.append(a)
        curr_r = (curr_r - a * curr_s) // p
        
    return digits

def padic_to_float(digits, p):
    """Convert p-adic digits to a float value (standard sum evaluation)."""
    val = 0.0
    for i, d in enumerate(digits):
        val += d * (p ** i)
    return val

def solve_poly_padic(coeffs, p, N):
    """
    Find all roots of a polynomial sum(coeffs[i] * x^i) = 0 in Z_p up to N digits.
    coeffs: list of coefficients [c_0, c_1, c_2, ...] representing c_0 + c_1*x + c_2*x^2 + ...
    Returns a list of lists, where each list is the N-digit p-adic representation of a root.
    """
    # 1. Find roots modulo p by brute force
    roots_mod_p = []
    for x in range(p):
        val = sum(c * (x ** i) for i, c in enumerate(coeffs)) % p
        if val == 0:
            # Check if derivative is non-zero mod p (Hensel's Lemma condition)
            deriv_val = sum(i * c * (x ** (i - 1)) for i, c in enumerate(coeffs) if i > 0) % p
            if deriv_val != 0:
                roots_mod_p.append((x, deriv_val))
                
    # 2. Lift each root using Hensel's Lemma
    all_roots_digits = []
    for x0, deriv_val in roots_mod_p:
        digits = [x0]
        curr_x = x0
        deriv_inv = mod_inverse(deriv_val, p)
        
        # We need N digits. We already have x_0 (digits[0]).
        # For n = 0 to N-2:
        # We want to find a_{n+1} such that x_{n+1} = x_n + a_{n+1} * p^(n+1)
        # a_{n+1} = - (f(x_n) / p^(n+1)) * [f'(x_0)]^-1 mod p
        for n in range(N - 1):
            f_val = sum(c * (int(curr_x) ** i) for i, c in enumerate(coeffs))
            # f_val must be divisible by p^(n+1)
            y = f_val // (p ** (n + 1))
            a_next = (-y * deriv_inv) % p
            digits.append(a_next)
            curr_x = curr_x + a_next * (p ** (n + 1))
            
        all_roots_digits.append(digits)
        
    return all_roots_digits

if __name__ == "__main__":
    # Self-test code
    print("Testing rational_to_padic:")
    # -1/3 in base 5
    digits = rational_to_padic(-1, 3, 5, 10)
    print("-1/3 mod 5^10:", digits)
    assert digits == [3, 1, 3, 1, 3, 1, 3, 1, 3, 1]
    
    # -1/3 in base 2
    digits2 = rational_to_padic(-1, 3, 2, 10)
    print("-1/3 mod 2^10:", digits2)
    assert digits2 == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    print("\nTesting solve_poly_padic (x^2 + 1 = 0 mod 5^10):")
    roots = solve_poly_padic([1, 0, 1], 5, 10)
    for idx, r in enumerate(roots):
        print(f"Root {idx}: {r}")
        # Verify: sum(r_i * 5^i)^2 + 1 should be 0 mod 5^10
        val = padic_to_float(r, 5)
        print(f"Verify root^2 + 1 mod 5^10: {int(val**2 + 1) % (5**10)}")
        assert int(val**2 + 1) % (5**10) == 0
    print("All tests passed!")
