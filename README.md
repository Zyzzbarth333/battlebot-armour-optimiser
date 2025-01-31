# BattleBot Armour Optimisation System

## Overview
This Python module provides a sophisticated tool for optimising multi-layered armour configurations in combat robots. It analyses different material combinations and layer arrangements to find optimal protection against various types of impacts while considering weight and cost constraints.

## Features
- Multi-layer armour configuration optimisation
- Material synergy calculations
- Layer thickness optimisation
- Impact-specific protection analysis (hammer, spinner, saw)
- Weight and cost consideration
- Customisable material properties database

## Technical Details

### Materials
The system includes a database of common engineering materials with their physical properties:

| Material | Grade/Type | Density (kg/m³) | Yield Strength (MPa) | Elastic Modulus (GPa) | Hardness (Brinell) | Cost (per kg) | Machinability (0-100) |
|----------|------------|----------------|---------------------|---------------------|------------------|---------------|---------------------|
| HDPE     | Standard   | 970           | 26                  | 0.8                 | 60               | 2.5           | 90                  |
| TPU      | Standard   | 1210          | 35                  | 0.03                | 80               | 3.0           | 85                  |
| Aluminium| 6061       | 2700          | 276                 | 69                  | 95               | 5.0           | 80                  |
| Steel    | 4140       | 7850          | 655                 | 200                 | 197              | 4.0           | 60                  |
| Titanium | Ti-6Al-4V  | 4430          | 880                 | 114                 | 334              | 8.0           | 55                  |

Notes:
- Cost is in relative units
- Machinability is rated on a scale of 0-100, where higher values indicate easier machining
- Material grades are specified for metals to ensure precise property references

### Impact Types
The system optimises for three types of impacts:
1. **Hammer** - Emphasises elastic properties (0.4) over hardness (0.3) and yield strength (0.3)
2. **Spinner** - Prioritises hardness (0.5) over yield strength (0.3) and elastic properties (0.2)
3. **Saw** - Heavily weights hardness (0.6) with lower emphasis on yield strength (0.3) and elastic properties (0.1)

### Layer Configuration
- Supports 2 or 3 layer configurations
- Minimum thickness of 1mm per layer
- Thicknesses are calculated in 1mm increments
- Position-based effectiveness multipliers (outer layers more significant)

### Weight Calculation
Weights are calculated using:
```python
weight = density * thickness_in_meters  # Results in kg/m²
```

## Usage

### Basic Usage

```python
from armour_optimiser import ArmourOptimiser, analyse_armour_options

# Create optimiser instance
optimiser = ArmourOptimiser()

# Analyse options for specific parameters
analyse_armour_options(
    optimiser,
    total_thickness=6.0,  # mm
    impact_type='hammer',
    max_weight=47.1  # kg/m²
)
```

### Example Output
```
Analysing armour options for hammer impacts:
Total thickness: 6.0mm
Maximum weight limit: 47.1 kg/m²

Top 2-layer configurations:
Option 1:
- Aluminium: 1.0mm
- Titanium: 5.0mm
Protection Score: 0.4
Total Weight: 24.9 kg/m²
Total Cost: 190.7 units
Synergy Bonus: 1.00x

Top 3-layer configurations:
Option 1:
- Aluminium: 1.0mm
- Steel: 1.0mm
- Titanium: 4.0mm
Protection Score: 0.3
Total Weight: 28.3 kg/m²
Total Cost: 186.7 units
Synergy Bonus: 1.00x
```

The examples above demonstrate how the system optimises for both 2 and 3 layer configurations. Note how in the 3-layer configuration, the system places progressively denser and harder materials (Aluminium → Steel → Titanium) from outer to inner layers, while maintaining minimum thickness requirements.

### Customising Materials
New materials can be added to the database in the `ArmourOptimiser` class:
```python
self.materials = {
    'CustomMaterial': Material(
        name='Custom',
        density=1000,           # kg/m³
        yield_strength=100,     # MPa
        elastic_modulus=50,     # GPa
        hardness=150,           # Brinell
        cost_per_kg=3.0,        # Relative units
        machinability=75        # Scale 0-100
    )
}
```

## Protection Score Calculation
The protection score for each layer is calculated using:
```python
layer_protection = (
    hardness_coeff * material.hardness * thickness +
    elastic_coeff * material.elastic_modulus * thickness +
    yield_coeff * material.yield_strength * thickness
) / material.density
```

Position multipliers are then applied:
- First layer: 1.0
- Second layer: 0.8
- Third layer: 0.6

## Limitations and Considerations
- Synergy effects between materials are currently set to 1.0 (neutral)
- Cost calculations are based on relative units
- The system assumes uniform impact distribution
- Edge effects and joining methods are not considered
- Material properties are simplified and may not reflect all real-world behaviors

## Future Improvements
- Implementation of realistic material synergy effects
- Addition of more impact types
- Consideration of temperature effects
- Integration of manufacturing constraints
- Support for more than 3 layers
- Advanced cost modeling including manufacturing costs
- Integration of finite element analysis for more accurate predictions

## Requirements
- Python 3.6+
- NumPy
- Dataclasses (included in Python 3.7+)

## Installation
Clone the repository and install dependencies:
```bash
git clone [repository-url]
cd battlebot-armour-optimiser
pip install -r requirements.txt
```

## License
