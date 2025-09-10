#!/usr/bin/env python3
"""
Debug script to investigate convergence issues in get_iterative_values function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.calculations import get_iterative_values, get_lc, get_qtn

def debug_get_iterative_values(qt_value, sig_v0_value, sig_v0_prime_value, fr_percent_value, max_debug_iterations=200):
    """
    Debug version of get_iterative_values that shows convergence behavior
    """
    print(f"\n=== DEBUGGING CONVERGENCE ===")
    print(f"Input values:")
    print(f"  qt = {qt_value}")
    print(f"  sig_v0 = {sig_v0_value}")
    print(f"  sig_v0_prime = {sig_v0_prime_value}")
    print(f"  fr_percent = {fr_percent_value}")
    print()
    
    n = 0.0
    lc = 0.0 if fr_percent_value == 0 else get_lc(qt_value, fr_percent_value)
    ntrial = 0.381*lc + 0.05*(sig_v0_prime_value/100)-0.15
    err = abs(ntrial - n)
    qtn_val = 0
    
    print(f"Initial values:")
    print(f"  lc = {lc}")
    print(f"  ntrial = {ntrial}")
    print(f"  err = {err}")
    print()
    
    iterations = 0
    print("Iteration details:")
    print("Iter | qtn_val  | lc       | n        | err      | ntrial")
    print("-" * 55)
    
    while err > 0.001 and iterations < max_debug_iterations:
        qtn_val = get_qtn(qt_value, sig_v0_value, sig_v0_prime_value, ntrial)
        lc = 0.0 if fr_percent_value == 0 else get_lc(qtn_val, fr_percent_value)
        n = min(1, 0.381*lc + 0.05*(sig_v0_prime_value/100)-0.15)
        err = abs(ntrial - n)
        
        # Print every 10th iteration, first 5, and last few
        if iterations < 5 or iterations % 10 == 0 or err <= 0.001 or iterations >= max_debug_iterations - 1:
            print(f"{iterations:4d} | {qtn_val:8.4f} | {lc:8.4f} | {n:8.4f} | {err:8.6f} | {ntrial:8.4f}")
        
        ntrial = n
        iterations += 1
        
        # Check for oscillation
        if iterations > 50 and err > 0.1:
            print(f"WARNING: Possible oscillation detected at iteration {iterations}")
            break
    
    print("-" * 55)
    print(f"Final result after {iterations} iterations:")
    print(f"  qtn = {qtn_val}")
    print(f"  lc = {lc}")
    print(f"  n = {n}")
    print(f"  final_error = {err}")
    print(f"  converged = {err <= 0.001}")
    print()
    
    return {'qtn': qtn_val, 'lc': lc, 'n': n, 'iterations': iterations, 'converged': err <= 0.001}

def test_korvest_values():
    """Test with values that might be in the Korvest file"""
    print("TESTING KORVEST-LIKE VALUES")
    print("=" * 50)
    
    # Test with the values from test_coarse_spacing.csv (which might represent Korvest data)
    test_cases = [
        # (qt, sig_v0, sig_v0_prime, fr_percent)
        (150.5, 19.5, 19.5, 4.6/150.5),  # First row from test file
        (765.2, 39.0, 39.0, 7.5/765.2),  # Second row
        (114.8, 58.5, 58.5, 1.2/114.8),  # Third row
        (1507.3, 78.0, 78.0, 18.0/1507.3),  # Fourth row
        (1503.1, 97.5, 97.5, 23.4/1503.1),  # Fifth row
        
        # Test some extreme values that might cause issues
        (0.1, 1.0, 1.0, 0.1),  # Very low qt
        (10000, 1000, 1000, 0.001),  # Very high qt, low fr
        (1.0, 0.1, 0.1, 10.0),  # High fr relative to qt
        (100, 100, 0.01, 1.0),  # Very low effective stress
    ]
    
    problematic_cases = []
    
    for i, (qt, sig_v0, sig_v0_prime, fr_percent) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        try:
            result = debug_get_iterative_values(qt, sig_v0, sig_v0_prime, fr_percent)
            if not result['converged'] or result['iterations'] > 50:
                problematic_cases.append((i+1, qt, sig_v0, sig_v0_prime, fr_percent, result))
        except Exception as e:
            print(f"ERROR in test case {i+1}: {e}")
            problematic_cases.append((i+1, qt, sig_v0, sig_v0_prime, fr_percent, {'error': str(e)}))
    
    print("\n" + "=" * 50)
    print("SUMMARY OF PROBLEMATIC CASES:")
    print("=" * 50)
    
    if problematic_cases:
        for case_num, qt, sig_v0, sig_v0_prime, fr_percent, result in problematic_cases:
            print(f"\nCase {case_num}: qt={qt}, sig_v0={sig_v0}, sig_v0_prime={sig_v0_prime}, fr_percent={fr_percent:.6f}")
            if 'error' in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Iterations: {result.get('iterations', 'N/A')}")
                print(f"  Converged: {result.get('converged', False)}")
    else:
        print("No problematic cases found!")

def analyze_convergence_math():
    """Analyze the mathematical behavior of the convergence algorithm"""
    print("\n" + "=" * 50)
    print("MATHEMATICAL ANALYSIS")
    print("=" * 50)
    
    print("\nThe convergence algorithm:")
    print("1. lc = get_lc(qtn, fr_percent)")
    print("2. n = min(1, 0.381*lc + 0.05*(sig_v0_prime/100) - 0.15)")
    print("3. qtn = get_qtn(qt, sig_v0, sig_v0_prime, n)")
    print("4. Repeat until |n_new - n_old| < 0.001")
    print()
    print("Potential issues:")
    print("- If lc calculation is unstable")
    print("- If the equation for n causes oscillation")
    print("- If qtn calculation has numerical issues")
    print("- If the system has multiple equilibrium points")

if __name__ == "__main__":
    print("CONVERGENCE INVESTIGATION TOOL")
    print("=" * 50)
    
    test_korvest_values()
    analyze_convergence_math()
    
    print("\n" + "=" * 50)
    print("INVESTIGATION COMPLETE")
    print("=" * 50)
