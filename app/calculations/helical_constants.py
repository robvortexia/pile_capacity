import math

def calculate_helical_constants(pile_params):
    """
    Calculate basic geometric constants for helical pile analysis
    
    Args:
        pile_params (dict): Dictionary containing pile parameters including:
            - shaft_diameter (float): Diameter of pile shaft in meters
            - helix_diameter (float): Diameter of helical plate in meters
    
    Returns:
        dict: Dictionary containing:
            - perimeter (float): Shaft perimeter in meters
            - helix_area (float): Area of helical plate in square meters
    """
    
    # Extract parameters
    shaft_diameter = pile_params['shaft_diameter']
    helix_diameter = pile_params['helix_diameter']
    
    # Calculate perimeter of shaft
    perimeter = math.pi * shaft_diameter
    
    # Calculate area of helical plate
    helix_area = math.pi * (helix_diameter ** 2) * 0.25
    
    return {
        'perimeter': perimeter,
        'helix_area': helix_area
    } 