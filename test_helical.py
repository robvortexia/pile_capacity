import json
import logging
from app.helical_calculations import calculate_helical_pile_results
from app.calculations import pre_input_calc  # Import your existing CPT processing function

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Sample CPT data
sample_cpt_data = [
    {'z': 0.5, 'qc': 2.5, 'fs': 0.05, 'gtot': 18.0},
    {'z': 1.0, 'qc': 3.0, 'fs': 0.06, 'gtot': 18.5},
    {'z': 1.5, 'qc': 3.5, 'fs': 0.07, 'gtot': 19.0},
    {'z': 2.0, 'qc': 4.0, 'fs': 0.08, 'gtot': 19.0},
    {'z': 2.5, 'qc': 4.5, 'fs': 0.09, 'gtot': 19.0},
    {'z': 3.0, 'qc': 5.0, 'fs': 0.10, 'gtot': 19.0},
    {'z': 3.5, 'qc': 5.5, 'fs': 0.11, 'gtot': 19.0},
    {'z': 4.0, 'qc': 6.0, 'fs': 0.12, 'gtot': 19.0},
    {'z': 4.5, 'qc': 6.5, 'fs': 0.13, 'gtot': 19.0},
    {'z': 5.0, 'qc': 7.0, 'fs': 0.14, 'gtot': 19.0},
    {'z': 5.5, 'qc': 7.5, 'fs': 0.15, 'gtot': 19.0},
    {'z': 6.0, 'qc': 8.0, 'fs': 0.16, 'gtot': 19.0},
    {'z': 6.5, 'qc': 8.5, 'fs': 0.17, 'gtot': 19.0},
    {'z': 7.0, 'qc': 9.0, 'fs': 0.18, 'gtot': 19.0},
    {'z': 7.5, 'qc': 9.5, 'fs': 0.19, 'gtot': 19.0},
    {'z': 8.0, 'qc': 10.0, 'fs': 0.20, 'gtot': 19.0},
]

# Sample pile parameters
pile_params = {
    'shaft_diameter': 0.762,
    'helix_diameter': 1.5,
    'helix_depth': 7.0,
    'borehole_depth': 1.0,
    'water_table': 2.0
}

def run_test():
    try:
        logger.info("Testing helical pile calculations")
        
        # Process CPT data
        processed_cpt = pre_input_calc({'cpt_data': sample_cpt_data}, pile_params['water_table'])
        
        if not processed_cpt:
            logger.error("Failed to process CPT data")
            return
        
        # Calculate helical pile results
        results = calculate_helical_pile_results(processed_cpt, pile_params)
        
        # Print results
        logger.info("Calculation completed successfully")
        logger.info(f"Summary results: {json.dumps(results['summary'], indent=2)}")
        logger.info("First few items from detailed results:")
        
        # Print a subset of detailed results to avoid overwhelming the console
        for key, value in results['detailed'].items():
            if isinstance(value, list) and len(value) > 3:
                logger.info(f"{key}: {value[:3]} ... (total: {len(value)} items)")
            else:
                logger.info(f"{key}: {value}")
                
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    run_test() 