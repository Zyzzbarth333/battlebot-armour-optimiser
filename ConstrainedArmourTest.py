from armour_optimiser import ArmourOptimiser, Material
import numpy as np
from itertools import combinations, permutations
from dataclasses import dataclass


@dataclass
class BotSurfaceAreas:
    """Stores the surface areas of each side of the BattleBot in m²"""
    left: float
    right: float
    front: float
    back: float
    top: float
    bottom: float


class ConstrainedArmourTest:
    def __init__(self, surface_areas: BotSurfaceAreas, max_total_weight: float):
        """
        Initialize with surface areas and maximum total weight.

        Args:
            surface_areas: BotSurfaceAreas object with surface areas in m²
            max_total_weight: Maximum allowed weight in kg
        """
        self.optimiser = ArmourOptimiser()
        self.surface_areas = {
            'left': surface_areas.left,
            'right': surface_areas.right,
            'front': surface_areas.front,
            'back': surface_areas.back,
            'top': surface_areas.top,
            'bottom': surface_areas.bottom
        }
        self.max_total_weight = max_total_weight
        # Base weight (before armor) - assume 60% of max weight for internal components
        self.base_weight = max_total_weight * 0.6
        self.remaining_weight = max_total_weight - self.base_weight

    def calculate_armor_weight(self, config: dict, side: str) -> float:
        """Calculate weight of armor configuration for given surface area"""
        area = self.surface_areas[side]
        total_weight = 0
        for material_name, thickness in zip(config['materials'], config['thicknesses']):
            material = self.optimiser.materials[material_name]
            weight = material.calculate_weight(thickness) * area  # kg/m² * m² = kg
            total_weight += weight
        return total_weight

    def get_valid_thicknesses(self, material_name: str, remaining_thickness: float) -> list:
        """Returns valid thicknesses for each material based on constraints."""
        if material_name in ['HDPE', 'TPU']:
            return np.arange(0.5, remaining_thickness + 0.5, 0.5).tolist()
        elif material_name == 'Aluminium':
            return np.arange(0.5, remaining_thickness + 0.5, 0.5).tolist()
        elif material_name == 'Steel':
            return np.arange(1.0, remaining_thickness + 0.5, 0.5).tolist()
        elif material_name == 'Titanium':
            return np.arange(0.5, min(1.0 + 0.5, remaining_thickness + 0.5), 0.5).tolist()
        return []

    def generate_valid_configs(self, num_layers: int, max_thickness: float) -> list:
        """Generate valid layer configurations based on constraints."""
        valid_configs = []
        material_names = list(self.optimiser.materials.keys())

        for base_combo in combinations(material_names, num_layers):
            for material_combo in permutations(base_combo):
                thickness_combinations = self._generate_valid_thickness_combinations(
                    material_combo,
                    max_thickness,
                    [],
                    max_thickness
                )

                for thicknesses in thickness_combinations:
                    if sum(thicknesses) <= max_thickness:
                        config = {
                            'materials': material_combo,
                            'thicknesses': thicknesses
                        }

                        # Check if this config would exceed total weight limit
                        total_armor_weight = sum(
                            self.calculate_armor_weight(config, side)
                            for side in self.surface_areas.keys()
                        )

                        if total_armor_weight <= self.remaining_weight:
                            valid_configs.append(config)

        return valid_configs

    def _generate_valid_thickness_combinations(self, materials: tuple, max_thickness: float,
                                               current_thicknesses: list, remaining_thickness: float) -> list:
        """Recursively generate valid thickness combinations for given materials."""
        if len(current_thicknesses) == len(materials):
            return [current_thicknesses] if sum(current_thicknesses) <= max_thickness else []

        current_material = materials[len(current_thicknesses)]
        valid_thicknesses = self.get_valid_thicknesses(current_material, remaining_thickness)

        combinations = []
        for thickness in valid_thicknesses:
            new_remaining = remaining_thickness - thickness
            if new_remaining >= 0:
                new_thicknesses = current_thicknesses + [thickness]
                combinations.extend(
                    self._generate_valid_thickness_combinations(
                        materials, max_thickness, new_thicknesses, new_remaining
                    )
                )
        return combinations

    def run_constrained_test(self, max_thickness: float = 6.0, impact_types: list = None):
        """Runs optimization with material-specific thickness constraints."""
        if impact_types is None:
            impact_types = ['hammer', 'spinner', 'saw']

        all_results = {impact_type: [] for impact_type in impact_types}

        for num_layers in [2, 3]:
            print(f"\nGenerating {num_layers}-layer configurations...")
            valid_configs = self.generate_valid_configs(num_layers, max_thickness)
            print(f"Found {len(valid_configs)} valid configurations")

            for impact_type in impact_types:
                print(f"Testing configurations for {impact_type} impacts...")

                for config in valid_configs:
                    materials = [self.optimiser.materials[name] for name in config['materials']]
                    effectiveness = self.optimiser.calculate_layer_effectiveness(
                        materials,
                        config['thicknesses'],
                        impact_type
                    )

                    result = {
                        'materials': config['materials'],
                        'thicknesses': config['thicknesses'],
                        'effectiveness': effectiveness,
                        'total_armor_weight': sum(
                            self.calculate_armor_weight(config, side)
                            for side in self.surface_areas.keys()
                        )
                    }
                    all_results[impact_type].append(result)

        self.display_results(all_results, impact_types)

    def display_results(self, all_results: dict, impact_types: list):
        """Displays the filtered and sorted results."""
        print("\nCONSTRAINED ARMOR CONFIGURATIONS")
        print("===============================")

        for impact_type in impact_types:
            print(f"\n{'=' * 60}")
            print(f"TOP 5 FOR {impact_type.upper()} IMPACTS")
            print(f"{'=' * 60}")

            type_results = sorted(
                all_results[impact_type],
                key=lambda x: x['effectiveness']['protection_score'],
                reverse=True
            )[:5]

            if not type_results:
                print("\nNo valid configurations found for this impact type.")
                continue

            for rank, config in enumerate(type_results, 1):
                print(f"\nRank {rank}:")
                print(f"Total Thickness: {sum(config['thicknesses']):.1f}mm")
                print(f"Number of Layers: {len(config['materials'])}")

                num_layers = len(config['materials'])
                for i, (material, thickness) in enumerate(zip(config['materials'], config['thicknesses'])):
                    if i == 0:
                        print(f"Outer Layer (Impact): {material} ({thickness:.1f}mm)")
                    elif i == num_layers - 1:
                        print(f"Inner Layer: {material} ({thickness:.1f}mm)")
                    else:
                        print(f"Layer {i}: {material} ({thickness:.1f}mm)")

                print(f"Protection Score: {config['effectiveness']['protection_score']:.1f}")
                print(f"Total Armor Weight: {config['total_armor_weight']:.1f} kg")
                print(f"Total Bot Weight: {config['total_armor_weight'] + self.base_weight:.1f} kg")
                print(f"Total Cost: {config['effectiveness']['total_cost']:.1f} units")
                print(f"Synergy Bonus: {config['effectiveness']['synergy_bonus']:.2f}x")

                # Print weight distribution
                print("\nWeight Distribution:")
                for side in self.surface_areas.keys():
                    weight = self.calculate_armor_weight(config, side)
                    print(f"  {side.capitalize()}: {weight:.1f} kg ({self.surface_areas[side]:.2f}m²)")


if __name__ == "__main__":
    # Example usage with a 13.61kg weight limit bot
    surface_areas = BotSurfaceAreas(
        left=0.045,  # 0.30m x 0.15m = 0.045m²
        right=0.045,  # 0.30m x 0.15m = 0.045m²
        front=0.030,  # 0.20m x 0.15m = 0.030m²
        back=0.030,  # 0.20m x 0.15m = 0.030m²
        top=0.060,  # 0.30m x 0.20m = 0.060m²
        bottom=0.060  # 0.30m x 0.20m = 0.060m²
    )

    # Create and run constrained test with 250kg weight limit
    tester = ConstrainedArmourTest(
        surface_areas=surface_areas,
        max_total_weight=13.61  # kg
    )

    print("Running Constrained Armor Analysis")
    print("=================================")
    print(f"Maximum Total Weight: {tester.max_total_weight:.1f}kg")
    print(f"Base Weight (internal components): {tester.base_weight:.1f}kg")
    print(f"Available Weight for Armor: {tester.remaining_weight:.1f}kg")
    print("\nSurface Areas:")
    for side, area in tester.surface_areas.items():
        print(f"{side.capitalize()}: {area:.2f}m²")

    # Run test with constraints
    tester.run_constrained_test(
        max_thickness=6.0,
        impact_types=['hammer', 'spinner', 'saw']
    )
