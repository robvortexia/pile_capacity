#!/usr/bin/env python3
"""
Debug script to test the actual processing pipeline with edge cases
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.calculations import get_iterative_values, pre_input_calc, get_fr_percent

def create_test_data_extreme_cases():
    """Create test data that might cause convergence issues"""
    
    # Test cases that might cause problems
    test_datasets = []
    
    # Case 1: Very fine data with extreme values
    case1 = []
    for i in range(100):  # 100 data points
        depth = i * 0.01  # 0.01m spacing like Korvest
        # Create some extreme values that might cause issues
        if i < 10:
            qc = 0.001  # Very low qc
            fs = 0.001  # Very low fs  
        elif i < 20:
            qc = 10000  # Very high qc
            fs = 1  # Low fs relative to qc
        elif i < 30:
            qc = 1  # Moderate qc
            fs = 1000  # Very high fs relative to qc
        else:
            qc = 100 + i  # Normal increasing values
            fs = 10 + i * 0.1
        
        gtot = 18.0  # Standard unit weight
        case1.append({'z': depth, 'qc': qc, 'fs': fs, 'gtot': gtot})
    
    test_datasets.append(("Extreme Values", case1))
    
    # Case 2: Zero or near-zero values
    case2 = []
    for i in range(50):
        depth = i * 0.02
        qc = 0.0001 if i % 5 == 0 else 1.0  # Some near-zero values
        fs = 0.0001 if i % 7 == 0 else 1.0
        gtot = 18.0
        case2.append({'z': depth, 'qc': qc, 'fs': fs, 'gtot': gtot})
    
    test_datasets.append(("Near-Zero Values", case2))
    
    # Case 3: Oscillating values
    case3 = []
    for i in range(100):
        depth = i * 0.01
        qc = 100 + 50 * np.sin(i * 0.1)  # Oscillating qc
        fs = 10 + 5 * np.cos(i * 0.1)    # Oscillating fs
        gtot = 18.0
        case3.append({'z': depth, 'qc': qc, 'fs': fs, 'gtot': gtot})
    
    test_datasets.append(("Oscillating Values", case3))
    
    # Case 4: Values that create problematic fr_percent
    case4 = []
    for i in range(50):
        depth = i * 0.02
        qc = 1.0  # Fixed low qc
        fs = 1000  # High fs - creates very high fr_percent
        gtot = 18.0
        case4.append({'z': depth, 'qc': qc, 'fs': fs, 'gtot': gtot})
    
    test_datasets.append(("High Fr Percent", case4))
    
    return test_datasets

def test_processing_pipeline(data, name, water_table=0.0):
    """Test the complete processing pipeline"""
    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print(f"{'='*60}")
    print(f"Data points: {len(data)}")
    print(f"Water table: {water_table}")
    
    # Create data structure like the real application
    test_data = {'cpt_data': data, 'water_table': water_table}
    
    try:
        # This is where the timeout would occur
        print("Running pre_input_calc...")
        result = pre_input_calc(test_data, water_table)
        
        if result is None:
            print("❌ pre_input_calc returned None")
            return False
        
        print("✅ pre_input_calc completed successfully")
        print(f"Processed {len(result['depth'])} data points")
        
        # Check for any problematic values in the results
        problematic_points = []
        for i in range(len(result['depth'])):
            if np.isnan(result['qtn'][i]) or np.isinf(result['qtn'][i]):
                problematic_points.append(i)
            if np.isnan(result['lc'][i]) or np.isinf(result['lc'][i]):
                problematic_points.append(i)
        
        if problematic_points:
            print(f"⚠️  Found {len(problematic_points)} problematic points: {problematic_points[:10]}...")
        else:
            print("✅ All calculated values are finite")
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR in processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def find_problematic_individual_values():
    """Test individual problematic values that might cause infinite loops"""
    print(f"\n{'='*60}")
    print("TESTING INDIVIDUAL PROBLEMATIC VALUES")
    print(f"{'='*60}")
    
    # Test cases that might cause infinite loops
    problematic_cases = [
        # (qt, sig_v0, sig_v0_prime, fr_percent, description)
        (0.0, 1.0, 1.0, 0.0, "All zeros"),
        (1e-10, 1e-10, 1e-10, 1e10, "Extreme small/large values"),
        (float('inf'), 1.0, 1.0, 1.0, "Infinite qt"),
        (1.0, float('inf'), float('inf'), 1.0, "Infinite stress"),
        (1.0, 1.0, 1.0, float('inf'), "Infinite fr_percent"),
        (float('nan'), 1.0, 1.0, 1.0, "NaN qt"),
        (-1.0, 1.0, 1.0, 1.0, "Negative qt"),
        (1.0, -1.0, 1.0, 1.0, "Negative sig_v0"),
        (1.0, 1.0, -1.0, 1.0, "Negative sig_v0_prime"),
    ]
    
    infinite_loop_cases = []
    
    for qt, sig_v0, sig_v0_prime, fr_percent, description in problematic_cases:
        print(f"\nTesting: {description}")
        print(f"  qt={qt}, sig_v0={sig_v0}, sig_v0_prime={sig_v0_prime}, fr_percent={fr_percent}")
        
        try:
            # Test with a timeout-like approach
            result = get_iterative_values(qt, sig_v0, sig_v0_prime, fr_percent)
            print(f"  ✅ Converged: qtn={result['qtn']:.4f}, lc={result['lc']:.4f}, n={result['n']:.4f}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            infinite_loop_cases.append((description, qt, sig_v0, sig_v0_prime, fr_percent, str(e)))
    
    if infinite_loop_cases:
        print(f"\n⚠️  Found {len(infinite_loop_cases)} cases that cause errors:")
        for desc, qt, sig_v0, sig_v0_prime, fr_percent, error in infinite_loop_cases:
            print(f"  - {desc}: {error}")
    else:
        print("\n✅ No infinite loop cases found in individual tests")

def test_with_original_algorithm():
    """Test what would happen with the original algorithm (no iteration limit)"""
    print(f"\n{'='*60}")
    print("SIMULATING ORIGINAL ALGORITHM BEHAVIOR")
    print(f"{'='*60}")
    
    # Test a case that might cause slow convergence
    qt, sig_v0, sig_v0_prime = 0.001, 0.001, 0.001  # Very small values
    fr_percent = 1000.0  # Very large fr_percent
    
    print(f"Testing with qt={qt}, sig_v0={sig_v0}, sig_v0_prime={sig_v0_prime}, fr_percent={fr_percent}")
    
    # Manually implement the original algorithm with tracking
    from app.calculations import get_lc, get_qtn
    
    n = 0.0
    lc = 0.0 if fr_percent == 0 else get_lc(qt, fr_percent)
    ntrial = 0.381*lc + 0.05*(sig_v0_prime/100)-0.15
    err = abs(ntrial - n)
    qtn_val = 0
    
    max_test_iterations = 1000  # Test up to 1000 iterations
    iterations = 0
    last_errors = []
    
    print("Iteration tracking:")
    while err > 0.001 and iterations < max_test_iterations:
        qtn_val = get_qtn(qt, sig_v0, sig_v0_prime, ntrial)
        lc = 0.0 if fr_percent == 0 else get_lc(qtn_val, fr_percent)
        n = min(1, 0.381*lc + 0.05*(sig_v0_prime/100)-0.15)
        err = abs(ntrial - n)
        ntrial = n
        iterations += 1
        
        # Track error progression
        last_errors.append(err)
        if len(last_errors) > 10:
            last_errors.pop(0)
        
        # Print every 100 iterations
        if iterations % 100 == 0:
            print(f"  Iteration {iterations}: error = {err:.8f}")
            
        # Check for oscillation
        if len(last_errors) >= 10 and all(e > 0.01 for e in last_errors[-5:]):
            print(f"  ⚠️  Possible oscillation detected at iteration {iterations}")
            break
    
    print(f"\nResult after {iterations} iterations:")
    print(f"  Final error: {err}")
    print(f"  Would have caused timeout: {iterations >= max_test_iterations or err > 0.001}")

if __name__ == "__main__":
    print("COMPREHENSIVE CONVERGENCE INVESTIGATION")
    print("="*60)
    
    # Test 1: Process complete datasets
    test_datasets = create_test_data_extreme_cases()
    for name, data in test_datasets:
        success = test_processing_pipeline(data, name)
        if not success:
            print(f"❌ Dataset '{name}' failed processing")
    
    # Test 2: Individual problematic values
    find_problematic_individual_values()
    
    # Test 3: Simulate original algorithm
    test_with_original_algorithm()
    
    print(f"\n{'='*60}")
    print("INVESTIGATION COMPLETE")
    print(f"{'='*60}")
