import math
import random
import torch
from torch.utils.data import Dataset
from padic_math import rational_to_padic, solve_poly_padic

class PadicDataset(Dataset):
    def __init__(self, primes=[2, 3, 5, 7, 11], N=32, num_samples_per_type=1000):
        """
        PyTorch Dataset for p-adic integers across different primes.
        primes: list of prime bases to generate data for
        N: number of digits in the truncated p-adic expansion
        num_samples_per_type: number of samples to generate per prime and per category
        """
        self.primes = primes
        self.N = N
        self.samples = []
        
        print(f"Generating p-adic dataset (N={N}) for primes {primes}...")
        for p in primes:
            # 1. Generate Rationals
            rats = self._gen_rationals(p, N, num_samples_per_type)
            for r in rats:
                self.samples.append({
                    'digits': torch.tensor(r, dtype=torch.long),
                    'p': p,
                    'type': 0 # Rational
                })
            
            # 2. Generate Algebraic Roots
            algs = self._gen_algebraic(p, N, num_samples_per_type)
            for a in algs:
                self.samples.append({
                    'digits': torch.tensor(a, dtype=torch.long),
                    'p': p,
                    'type': 1 # Algebraic
                })
                
            # 3. Generate Random noise (as control/baseline)
            rands = self._gen_random(p, N, num_samples_per_type)
            for rn in rands:
                self.samples.append({
                    'digits': torch.tensor(rn, dtype=torch.long),
                    'p': p,
                    'type': 2 # Random
                })
            
            print(f"  Prime {p}: {len(rats)} rationals, {len(algs)} algebraic roots, {len(rands)} random sequences.")
        
        print(f"Total dataset size: {len(self.samples)} samples.")
        
    def _gen_rationals(self, p, N, count):
        rationals = []
        seen = set()
        attempts = 0
        while len(rationals) < count and attempts < 100000:
            attempts += 1
            s = random.randint(2, 500)
            if s % p == 0:
                continue
            r = random.randint(-s + 1, s - 1)
            if r == 0:
                continue
            # Simplify
            g = math.gcd(r, s)
            r, s = r // g, s // g
            
            if (r, s) in seen:
                continue
            seen.add((r, s))
            
            try:
                digits = rational_to_padic(r, s, p, N)
                rationals.append(digits)
            except Exception:
                pass
        return rationals
        
    def _gen_algebraic(self, p, N, count):
        algebraic = []
        seen = set()
        attempts = 0
        while len(algebraic) < count and attempts < 100000:
            attempts += 1
            # Sample random polynomial degree 2, 3, or 4
            deg = random.choice([2, 3, 4])
            coeffs = [random.randint(-150, 150) for _ in range(deg + 1)]
            if coeffs[-1] == 0:
                coeffs[-1] = 1
                
            coeffs_tuple = tuple(coeffs)
            if coeffs_tuple in seen:
                continue
            seen.add(coeffs_tuple)
            
            try:
                roots = solve_poly_padic(coeffs, p, N)
                for root in roots:
                    if len(algebraic) < count:
                        root_tuple = tuple(root)
                        if root_tuple not in seen:
                            seen.add(root_tuple)
                            algebraic.append(root)
            except Exception:
                pass
                
        # If we couldn't generate enough, print warning
        if len(algebraic) < count:
            print(f"  Warning: Only generated {len(algebraic)}/{count} algebraic roots for p={p}")
        return algebraic
        
    def _gen_random(self, p, N, count):
        rands = []
        for _ in range(count):
            rands.append([random.randint(0, p - 1) for _ in range(N)])
        return rands
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    # Test dataset creation
    ds = PadicDataset(primes=[2, 3, 5], N=32, num_samples_per_type=10)
    print("Dataset length:", len(ds))
    sample = ds[0]
    print("Sample digits:", sample['digits'])
    print("Sample prime p:", sample['p'])
    print("Sample type:", sample['type'])
