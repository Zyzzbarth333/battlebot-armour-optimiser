from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from itertools import combinations, permutations


@dataclass
class Material:
    """Material properties using SI units"""
    name: str
    density: float  # kg/m³
    yield_strength: float  # MPa
    ultimate_strength: float  # MPa
    elastic_modulus: float  # GPa
    hardness: float  # Brinell hardness
    cost_per_kg: float  # Relative cost units

    def get_weight_per_area(self, thickness_mm: float) -> float:
        """Calculate weight per square meter for given thickness"""
        return self.density * (thickness_mm / 1000.0)  # kg/m²


@dataclass
class BotSurfaceAreas:
    """Surface areas for each face of the robot in m²"""
    left: float
    right: float
    front: float
    back: float
    top: float
    bottom: float

    def get_surface_areas(self) -> Dict[str, float]:
        """Return surface areas as a dictionary"""
        return {
            'left': self.left,
            'right': self.right,
            'front': self.front,
            'back': self.back,
            'top': self.top,
            'bottom': self.bottom
        }


class ArmorTester:
    def __init__(self, surface_areas: BotSurfaceAreas, max_total_weight: float, max_layers: int = 3,
                 max_thickness: float = 6.0):
        """
        Initialize armor testing system

        Args:
            surface_areas: BotSurfaceAreas object
            max_total_weight: Maximum allowed weight in kg
            max_layers: Maximum number of layers to test (default: 5)
            max_thickness: Maximum total thickness in mm (default: 6.0)
        """
        self.materials = {
            'HDPE': Material('HDPE', 970, 26, 33, 0.8, 60, 2.5),
            'TPU': Material('TPU', 1210, 35, 45, 0.03, 80, 3.0),
            'Aluminum': Material('Aluminum 6061', 2700, 276, 310, 69, 95, 5.0),
            'Steel': Material('Steel 4140', 7850, 655, 1020, 200, 197, 4.0),
            'Titanium': Material('Titanium Ti-6Al-4V', 4430, 880, 950, 114, 334, 8.0)
        }

        self.surface_areas = surface_areas.get_surface_areas()
        self.max_total_weight = max_total_weight
        self.max_layers = max_layers
        self.max_thickness = max_thickness

        # Reserve 60% of weight for internals
        self.base_weight = max_total_weight * 0.6
        self.armor_weight_limit = max_total_weight - self.base_weight

        # Impact energies in Joules
        self.impact_energies = {
            'hammer': 5000.0,
            'spinner': 8000.0,
            'saw': 3000.0
        }

    def get_valid_thickness_range(self, material_name: str) -> Tuple[float, float, float]:
        """
        Get valid thickness range for a material

        Returns:
            Tuple of (min_thickness, max_thickness, step_size) in mm
        """
        # Basic thickness constraints
        if material_name in ['HDPE', 'TPU']:
            return (0.5, min(6.0, self.max_thickness), 0.5)
        elif material_name == 'Aluminum':
            return (0.5, min(2.0, self.max_thickness), 0.5)
        elif material_name == 'Steel':
            return (0.5, min(2.0, self.max_thickness), 0.5)
        elif material_name == 'Titanium':
            return (0.5, min(1.0, self.max_thickness), 0.5)
        return (0.5, min(6.0, self.max_thickness), 0.5)  # Default

    def calculate_config_weight(self,
                                materials: List[str],
                                thicknesses: List[float]) -> float:
        """Calculate total weight of armor configuration"""
        total_weight = 0.0
        for side_area in self.surface_areas.values():
            for material_name, thickness in zip(materials, thicknesses):
                material = self.materials[material_name]
                weight = material.get_weight_per_area(thickness) * side_area
                total_weight += weight
        return total_weight

    def simple_energy_absorption(self,
                                 material: Material,
                                 thickness_mm: float,
                                 impact_type: str) -> float:
        """
        Simplified initial energy absorption model
        This will be replaced with more sophisticated models later
        """
        # Convert to meters for calculation
        thickness_m = thickness_mm / 1000.0

        # Basic absorption proportional to material properties
        if impact_type == 'hammer':
            return material.yield_strength * thickness_m * 1000
        elif impact_type == 'spinner':
            return material.hardness * thickness_m * 1000
        else:  # saw
            return (material.hardness * 0.7 + material.yield_strength * 0.3) * thickness_m * 1000

    def evaluate_armor_config(self,
                              materials: List[str],
                              thicknesses: List[float],
                              impact_type: str) -> Dict:
        """
        Evaluate effectiveness of an armor configuration with physically accurate
        energy transfer. Each layer experiences the FULL remaining energy from
        the previous layer.
        """
        # Initial impact energy
        impact_energy = self.impact_energies[impact_type]
        current_energy = impact_energy  # Energy being transferred through layers

        total_weight = self.calculate_config_weight(materials, thicknesses)
        total_cost = 0.0

        layer_results = []

        # Process each layer
        for i, (material_name, thickness) in enumerate(zip(materials, thicknesses)):
            material = self.materials[material_name]

            # Calculate layer properties
            layer_weight_per_m2 = material.get_weight_per_area(thickness)
            layer_cost = sum(layer_weight_per_m2 * area * material.cost_per_kg
                             for area in self.surface_areas.values())
            total_cost += layer_cost

            # Calculate energy absorption capacity
            absorption_capacity = self.simple_energy_absorption(
                material, thickness, impact_type)

            # This layer experiences the full current_energy
            # But can only absorb up to its capacity
            energy_absorbed = min(absorption_capacity, current_energy)

            # Energy transferred to next layer is reduced by what was absorbed
            current_energy = max(0, current_energy - energy_absorbed)

            layer_results.append({
                'layer_index': i,
                'material': material_name,
                'thickness_mm': thickness,
                'absorption_capacity': absorption_capacity,
                'incoming_energy': current_energy + energy_absorbed,  # Energy this layer experienced
                'energy_absorbed': energy_absorbed,
                'energy_remaining': current_energy
            })

            if current_energy <= 0:
                break

        total_absorbed = impact_energy - current_energy
        protection_score = total_absorbed / impact_energy

        return {
            'protection_score': protection_score,
            'total_weight': total_weight,
            'total_cost': total_cost,
            'total_absorbed': total_absorbed,
            'remaining_energy': current_energy,
            'layer_breakdown': layer_results
        }

    def generate_valid_configs(self, num_layers: int = 2) -> List[Dict]:
        """Generate valid armor configurations"""
        valid_configs = []
        material_names = list(self.materials.keys())

        # Generate material combinations
        for material_combo in combinations(material_names, num_layers):
            # Try different arrangements of these materials
            for materials in permutations(material_combo):
                # Get valid thickness ranges for these materials
                thickness_ranges = [self.get_valid_thickness_range(m) for m in materials]

                # Generate thickness combinations
                for thicknesses in self._generate_thicknesses(thickness_ranges):
                    # Check weight constraint
                    total_weight = self.calculate_config_weight(materials, thicknesses)
                    if total_weight <= self.armor_weight_limit:
                        valid_configs.append({
                            'materials': list(materials),
                            'thicknesses': thicknesses
                        })

        return valid_configs

    def _generate_thicknesses(self, thickness_ranges: List[Tuple[float, float, float]]) -> List[List[float]]:
        """Generate valid thickness combinations recursively for any number of layers"""

        def recursive_thickness_gen(current_thicknesses: List[float],
                                    remaining_ranges: List[Tuple[float, float, float]],
                                    remaining_thickness: float) -> List[List[float]]:
            # Base case - if we've used all ranges, return the current combination if valid
            if not remaining_ranges:
                if current_thicknesses:  # Only return if we have thicknesses
                    return [current_thicknesses]
                return []

            min_t, max_t, step = remaining_ranges[0]
            combinations = []

            # Adjust maximum thickness based on what's remaining
            adjusted_max = min(max_t, remaining_thickness)

            # Try each possible thickness for this layer
            for thickness in np.arange(min_t, adjusted_max + step, step):
                if thickness <= remaining_thickness:
                    new_thicknesses = current_thicknesses + [thickness]
                    # Recursively generate combinations for remaining layers
                    combinations.extend(
                        recursive_thickness_gen(
                            new_thicknesses,
                            remaining_ranges[1:],
                            remaining_thickness - thickness
                        )
                    )

            return combinations

        # Start the recursive generation with maximum total thickness
        return recursive_thickness_gen([], thickness_ranges, self.max_thickness)

    def test_armor_configurations(self, impact_type: str):
        """Test all valid configurations for a given impact type"""
        results = []

        for num_layers in range(1, self.max_layers + 1):  # Changed this line
            print(f"\nGenerating {num_layers}-layer configurations...")
            configs = self.generate_valid_configs(num_layers)
            print(f"Found {len(configs)} valid configurations")

            for config in configs:
                effectiveness = self.evaluate_armor_config(
                    config['materials'],
                    config['thicknesses'],
                    impact_type
                )

                results.append({
                    'materials': config['materials'],
                    'thicknesses': config['thicknesses'],
                    'effectiveness': effectiveness
                })

        # Sort by protection score and return top 5
        return sorted(
            results,
            key=lambda x: x['effectiveness']['protection_score'],
            reverse=True
        )[:5]


def print_results(results: List[Dict], impact_type: str):
    """Print results in a readable format"""
    print(f"\n{'=' * 60}")
    print(f"TOP 5 FOR {impact_type.upper()} IMPACTS")
    print(f"{'=' * 60}")

    for rank, config in enumerate(results, 1):
        print(f"\nRank {rank}:")
        print(f"Total Thickness: {sum(config['thicknesses']):.1f} mm")

        for i, (material, thickness) in enumerate(zip(
                config['materials'], config['thicknesses'])):
            layer_type = "Outer" if i == 0 else "Inner" if i == len(
                config['materials']) - 1 else "Middle"
            print(f"{layer_type} Layer: {material} ({thickness:.1f} mm)")

        eff = config['effectiveness']
        print(f"Protection Score: {eff['protection_score']:.2f}")
        print(f"Total Absorbed Energy: {eff['total_absorbed']:.1f} J")
        print(f"Total Weight: {eff['total_weight']:.2f} kg/m²")
        print(f"Total Cost: {eff['total_cost']:.2f} units")

        print("\nLayer-by-Layer Breakdown:")
        for layer in eff['layer_breakdown']:
            print(f"  Layer {layer['layer_index']}: {layer['material']}")
            print(f"    Thickness: {layer['thickness_mm']:.1f} mm")
            print(f"    Absorption Capacity: {layer['absorption_capacity']:.1f} J")
            print(f"    Incoming Energy: {layer['incoming_energy']:.1f} J")  # Added this line
            print(f"    Energy Absorbed: {layer['energy_absorbed']:.1f} J")
            print(f"    Energy Remaining: {layer['energy_remaining']:.1f} J")

        print("-" * 60)


if __name__ == "__main__":
    # Example usage with explicit surface areas
    surface_areas = BotSurfaceAreas(
        left=0.045,  # 0.30m x 0.15m = 0.045m²
        right=0.045,  # 0.30m x 0.15m = 0.045m²
        front=0.030,  # 0.20m x 0.15m = 0.030m²
        back=0.030,  # 0.20m x 0.15m = 0.030m²
        top=0.060,  # 0.30m x 0.20m = 0.060m²
        bottom=0.060  # 0.30m x 0.20m = 0.060m²
    )

    # Create tester with 13.61kg weight limit
    # As well as the maximum amount of layers (1mm for each material)
    # Maximum thickness of armour
    tester = ArmorTester(
        surface_areas=surface_areas,
        max_total_weight=13.61,
        max_layers=5,
        max_thickness=6.0
    )

    print("Running Armor Analysis")
    print("=====================")
    print(f"Maximum Total Weight: {tester.max_total_weight:.1f}kg")
    print(f"Base Weight (internal components): {tester.base_weight:.1f}kg")
    print(f"Available Weight for Armor: {tester.armor_weight_limit:.1f}kg")

    print("\nSurface Areas:")
    for side, area in tester.surface_areas.items():
        print(f"{side.capitalize()}: {area:.3f}m²")

    # Test configurations
    for impact_type in ['hammer', 'spinner', 'saw']:
        results = tester.test_armor_configurations(impact_type)
        print_results(results, impact_type)
