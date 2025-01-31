"""
BattleBot Armour Optimisation System
=====================================

This module provides tools for optimising multi-layered armour configurations
for BattleBots. It analyses different material combinations and layer
arrangements to find optimal protection against various types of impacts.

Key Features:
- Multiple material support with synergy calculations
- Layer thickness optimisation
- Impact-specific protection analysis (blunt vs. penetration vs. abrasion)
- Weight and cost considerations
"""

import numpy as np
from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict


@dataclass
class Material:
    """
    Represents a material with its physical and economic properties.

    All measurements use SI units unless otherwise specified.
    """
    name: str
    density: float  # kg/m³
    yield_strength: float  # MPa
    elastic_modulus: float  # GPa
    hardness: float  # Brinell hardness
    cost_per_kg: float  # Relative cost units
    machinability: float  # Scale of 0-100

    def calculate_weight(self, thickness_mm: float) -> float:
        """
        Calculate weight for a given thickness.

        Args:
            thickness_mm: Thickness in millimeters

        Returns:
            float: Weight in kg/m² for the given thickness
        """
        return self.density * (thickness_mm * 0.001)  # kg/m³ * m = kg/m²


class ArmourOptimiser:
    """
    Optimises multi-layered armour configurations for BattleBots.

    Utilises material properties and synergy effects to determine optimal
    protection against various impact types whilst considering weight and
    cost constraints.
    """

    def __init__(self):
        # Enhanced material properties database with some common engineering materials
        self.materials = {
            'HDPE': Material('HDPE', 970, 26, 0.8, 60, 2.5, 90),
            'TPU': Material('TPU', 1210, 35, 0.03, 80, 3.0, 85),
            'Aluminium': Material('Aluminium 6061', 2700, 276, 69, 95, 5.0, 80),
            'Steel': Material('Steel 4140', 7850, 655, 200, 197, 4.0, 60),
            'Titanium': Material('Titanium Ti-6Al-4V', 4430, 880, 114, 334, 8.0, 55)
        }

        # Store the density of the heaviest material
        self.max_density = max(mat.density for mat in self.materials.values())

        # Hypothetical impact energies (Joules) for different weapon types
        self.impact_energies = {
            'hammer': 5000.0,   # Example: heavy blunt impact
            'spinner': 8000.0,  # Example: high-speed penetrative impact
            'saw': 3000.0       # Example: abrasive/cutting
        }

        # Example synergy matrix (all 1.0 = no extra synergy by default)
        self.synergy_matrix = {
            ('HDPE', 'TPU'): 1.0,
            ('HDPE', 'Aluminium'): 1.0,
            ('HDPE', 'Steel'): 1.0,
            ('HDPE', 'Titanium'): 1.0,

            ('TPU', 'HDPE'): 1.0,
            ('TPU', 'Aluminium'): 1.0,
            ('TPU', 'Steel'): 1.0,
            ('TPU', 'Titanium'): 1.0,

            ('Aluminium', 'HDPE'): 1.0,
            ('Aluminium', 'TPU'): 1.0,
            ('Aluminium', 'Steel'): 1.0,
            ('Aluminium', 'Titanium'): 1.0,

            ('Steel', 'HDPE'): 1.0,
            ('Steel', 'TPU'): 1.0,
            ('Steel', 'Aluminium'): 1.0,
            ('Steel', 'Titanium'): 1.0,

            ('Titanium', 'HDPE'): 1.0,
            ('Titanium', 'TPU'): 1.0,
            ('Titanium', 'Aluminium'): 1.0,
            ('Titanium', 'Steel'): 1.0
        }

    # -------------------------------------------------------------------------
    # 1) Separate "absorption" models for each type:
    # -------------------------------------------------------------------------

    def _absorb_blunt(self, material: Material, thickness_m: float) -> float:
        """
        Approximate capacity (in Joules) for a blunt (hammer-like) impact.

        Emphasizes yield strength (material's ductile/plastic absorption).
        """
        sigma_y_pa = material.yield_strength * 1e6  # MPa -> Pa
        # Base factor for converting (Pa * volume) into Joules
        base_factor = 1e-5
        # Simple proportion to yield strength * thickness
        capacity = sigma_y_pa * thickness_m * base_factor
        return capacity

    def _absorb_penetration(self, material: Material, thickness_m: float) -> float:
        """
        Approximate capacity (in Joules) for a penetrating (spinner-like) impact.

        Spinners often rely on local hardness and toughness to resist penetration.
        """
        # Hardness-based approach
        hardness_factor = (material.hardness / 100.0)
        base_factor = 1e-3

        capacity = hardness_factor * thickness_m * base_factor
        # Add a bit from the blunt formula
        capacity += 0.1 * self._absorb_blunt(material, thickness_m)
        return capacity

    def _absorb_abrasion(self, material: Material, thickness_m: float) -> float:
        """
        Approximate capacity (in Joules) for an abrasive/cutting (saw-like) impact.
        """
        # Heavier weighting on hardness, plus a minor yield component
        hardness_factor = (material.hardness / 10.0)
        base_factor = 1e-3

        capacity = hardness_factor * thickness_m * base_factor
        capacity += 0.05 * self._absorb_blunt(material, thickness_m)
        return capacity

    def _layer_absorption_capacity(self,
                                   material: Material,
                                   thickness_mm: float,
                                   impact_type: str) -> float:
        """
        Estimate how much impact energy (in Joules) a single layer can absorb
        based on thickness and material properties.

        We'll pick a different formula for each weapon type:
         - 'hammer'   -> blunt
         - 'spinner'  -> penetration
         - 'saw'      -> abrasion
        """
        thickness_m = thickness_mm / 1000.0

        if impact_type == 'hammer':
            return self._absorb_blunt(material, thickness_m)
        elif impact_type == 'spinner':
            return self._absorb_penetration(material, thickness_m)
        elif impact_type == 'saw':
            return self._absorb_abrasion(material, thickness_m)
        else:
            # Default if unknown
            return self._absorb_blunt(material, thickness_m)

    # -------------------------------------------------------------------------
    # 2) Overall layer-by-layer absorption logic:
    # -------------------------------------------------------------------------
    def calculate_layer_effectiveness(self,
                                      materials: List[Material],
                                      thicknesses: List[float],
                                      impact_type: str) -> Dict:
        """
        Calculate the effectiveness of a specific layer configuration using
        a simplified energy-absorption model:

        1. The total impact energy (for the given impact_type) hits
           the outermost layer first.
        2. Each layer tries to absorb up to its capacity.
        3. Leftover energy passes to the next layer.
        4. The final leftover after the last layer determines how much
           was not absorbed (so total absorbed = initial - leftover).

        Returns a dict with:
           - protection_score (0.0 to 1.0, fraction of total energy absorbed)
           - total_weight (kg/m²)
           - total_cost (units)
           - synergy_bonus (dimensionless multiplier)
           - total_absorbed_energy (Joules)
           - final_leftover_energy (Joules)
           - layer_breakdown (list of details for each layer)
        """
        initial_energy = self.impact_energies.get(impact_type, 5000.0)
        leftover_energy = initial_energy

        total_weight = 0.0
        total_cost = 0.0
        synergy_bonus = 1.0

        # Track per-layer absorption details
        layer_breakdown = []

        for i, (material, thickness) in enumerate(zip(materials, thicknesses)):
            # Update total weight/cost
            layer_weight = material.calculate_weight(thickness)
            total_weight += layer_weight
            total_cost += layer_weight * material.cost_per_kg

            # Update synergy if consecutive layers have known synergy
            if i > 0:
                pair = (materials[i - 1].name, material.name)
                if pair in self.synergy_matrix:
                    synergy_bonus *= self.synergy_matrix[pair]
                else:
                    rev_pair = (pair[1], pair[0])
                    if rev_pair in self.synergy_matrix:
                        synergy_bonus *= self.synergy_matrix[rev_pair]

            # Compute layer's base capacity
            raw_capacity = self._layer_absorption_capacity(material, thickness, impact_type)
            # Apply synergy bonus to that capacity if desired
            layer_capacity = raw_capacity * synergy_bonus

            if leftover_energy > layer_capacity:
                absorbed_here = layer_capacity
                leftover_energy -= layer_capacity
            else:
                absorbed_here = leftover_energy
                leftover_energy = 0.0

            layer_breakdown.append({
                'layer_index': i,
                'material': material.name,
                'thickness_mm': thickness,
                'raw_capacity_joules': raw_capacity,
                'synergy_bonus': synergy_bonus,
                'actual_capacity_joules': layer_capacity,
                'absorbed_joules': absorbed_here,
                'leftover_after': leftover_energy
            })

            # If leftover is zero, no need to check further layers
            if leftover_energy <= 0:
                break

        total_absorbed = initial_energy - leftover_energy
        protection_score = total_absorbed / initial_energy  # fraction

        return {
            'protection_score': protection_score,
            'total_weight': total_weight,
            'total_cost': total_cost,
            'synergy_bonus': synergy_bonus,
            'total_absorbed_energy': total_absorbed,
            'final_leftover_energy': leftover_energy,
            'layer_breakdown': layer_breakdown
        }

    # -------------------------------------------------------------------------
    # 3) Generating layer thickness combos:
    # -------------------------------------------------------------------------
    def _generate_thickness_ratios(self,
                                   num_layers: int,
                                   total_thickness: float,
                                   thickness_step: float) -> List[List[float]]:
        """
        Generate different thickness ratio combinations for layers.
        Each layer must be at least 1mm thick and be in specified step increments.
        """
        thickness_combinations = []

        total_steps = int(total_thickness / thickness_step)
        min_steps = int(1.0 / thickness_step)  # minimum 1mm in steps

        if num_layers == 2:
            for i in range(min_steps, total_steps - min_steps + 1):
                t1 = i * thickness_step
                t2 = total_thickness - t1
                if t2 >= 1.0:
                    thickness_combinations.append([t1, t2])

        elif num_layers == 3:
            for i in range(min_steps, total_steps - (2 * min_steps) + 1):
                t1 = i * thickness_step
                remaining = total_thickness - t1
                for j in range(min_steps, int(remaining / thickness_step) - min_steps + 1):
                    t2 = j * thickness_step
                    t3 = remaining - t2
                    if t3 >= 1.0:
                        thickness_combinations.append([t1, t2, t3])

        return thickness_combinations

    # -------------------------------------------------------------------------
    # 4) Optimiser method to iterate over combos:
    # -------------------------------------------------------------------------
    def optimise_layers(self,
                        total_thickness: float,
                        num_layers: int,
                        impact_type: str,
                        max_weight: float,
                        thickness_step: float) -> List[dict]:
        """
        Find optimal layer configurations within given constraints.

        Args:
            total_thickness: Total armour thickness (mm)
            num_layers: Number of layers to use
            impact_type: Type of impact to optimise against
            max_weight: Maximum allowable weight in kg/m²
            thickness_step: Step size for thickness increments in mm

        Returns:
            List of top 5 configurations sorted by protection score
        """
        results = []
        material_names = list(self.materials.keys())

        # Generate all possible material combinations (no repetition)
        for material_combo in combinations(material_names, num_layers):
            # Try different thickness distributions
            for thickness_list in self._generate_thickness_ratios(num_layers,
                                                                  total_thickness,
                                                                  thickness_step):
                materials = [self.materials[name] for name in material_combo]

                # Evaluate this config
                effectiveness = self.calculate_layer_effectiveness(
                    materials,
                    thickness_list,
                    impact_type
                )

                # Check weight constraint
                if max_weight and effectiveness['total_weight'] > max_weight:
                    continue

                results.append({
                    'materials': material_combo,
                    'thicknesses': thickness_list,
                    'effectiveness': effectiveness
                })

        # Sort and return top 5
        return sorted(results,
                      key=lambda x: x['effectiveness']['protection_score'],
                      reverse=True)[:5]


def analyse_armour_options(optimiser: ArmourOptimiser,
                           total_thickness: float,
                           impact_type: str,
                           max_weight: float,
                           results_dict: dict,
                           thickness_step: float) -> dict:
    """
    Analyse and store armour options for given parameters.

    Args:
        optimiser: ArmourOptimiser instance
        total_thickness: Total thickness in millimetres
        impact_type: Type of impact to analyse
        max_weight: Maximum allowable weight in kg/m²
        results_dict: Dictionary to store results in
        thickness_step: Step size for thickness increments

    Returns:
        Dictionary containing results for each impact type
    """
    for num_layers in [2, 3]:
        configs = optimiser.optimise_layers(
            total_thickness,
            num_layers,
            impact_type,
            max_weight,
            thickness_step
        )
        if configs:
            for cfg in configs:
                cfg['total_thickness'] = total_thickness
            results_dict[impact_type].extend(configs)

    return results_dict


if __name__ == "__main__":
    # Create optimiser instance
    optimiser = ArmourOptimiser()

    print("BattleBots Armour Analysis")
    print("============================")

    # Analysis parameters
    MAX_THICKNESS = 6.0  # mm
    MIN_THICKNESS = 1.0  # mm
    THICKNESS_STEP = 0.5  # mm
    impact_types = ['hammer', 'spinner', 'saw']

    # Dictionary to collect results for all thicknesses & impact types
    all_results = {'hammer': [], 'spinner': [], 'saw': []}

    # Theoretical maximum weight for scaling
    max_possible_weight = optimiser.max_density * (MAX_THICKNESS * 0.001)

    # Try a range of total thicknesses
    for thickness in np.arange(MAX_THICKNESS, MIN_THICKNESS - THICKNESS_STEP, -THICKNESS_STEP):
        print(f"\n{'=' * 60}")
        print(f"Testing {thickness:.1f}mm Total Thickness Configurations")
        print(f"{'=' * 60}")

        # Weight limit for the current thickness
        weight_limit = optimiser.max_density * (thickness * 0.001)

        for impact_type in impact_types:
            all_results = analyse_armour_options(
                optimiser,
                thickness,
                impact_type,
                weight_limit,
                all_results,
                THICKNESS_STEP
            )

    # Print top 5 for each impact type from ALL results
    print("\n\nOVERALL BEST CONFIGURATIONS")
    print("===========================")

    for impact_type in impact_types:
        print(f"\n{'=' * 60}")
        print(f"TOP 5 FOR {impact_type.upper()} IMPACTS")
        print(f"{'=' * 60}")

        # Sort by protection score
        type_results = sorted(
            all_results[impact_type],
            key=lambda x: x['effectiveness']['protection_score'],
            reverse=True
        )[:5]

        for rank, config in enumerate(type_results, 1):
            print(f"\nRank {rank}:")
            print(f"Total Thickness: {config['total_thickness']:.1f} mm")

            # Print layers (outer, middle, inner)
            mats = list(zip(config['materials'], config['thicknesses']))
            if len(mats) == 2:
                print(f"  Outer Layer: {mats[0][0]} ({mats[0][1]:.1f} mm)")
                print(f"  Inner Layer: {mats[1][0]} ({mats[1][1]:.1f} mm)")
            else:  # 3 layers
                print(f"  Outer Layer: {mats[0][0]} ({mats[0][1]:.1f} mm)")
                print(f"  Middle Layer: {mats[1][0]} ({mats[1][1]:.1f} mm)")
                print(f"  Inner Layer: {mats[2][0]} ({mats[2][1]:.1f} mm)")

            eff = config['effectiveness']
            print(f"  Protection Score: {eff['protection_score']:.2f}")
            print(f"  Total Absorbed Energy: {eff['total_absorbed_energy']:.1f} J")
            print(f"  Total Weight: {eff['total_weight']:.2f} kg/m²")
            print(f"  Total Cost: {eff['total_cost']:.2f} units")
            print(f"  Synergy Bonus: {eff['synergy_bonus']:.2f}x")

            # -----------------------------------------------
            # HERE: Print the layer-by-layer breakdown
            # -----------------------------------------------
            print("\n  Layer-by-Layer Breakdown:")
            for lb in eff['layer_breakdown']:
                print(f"    Layer {lb['layer_index']}: {lb['material']}")
                print(f"      Thickness (mm): {lb['thickness_mm']:.2f}")
                print(f"      Raw Capacity (J): {lb['raw_capacity_joules']:.2f}")
                print(f"      Synergy Bonus: {lb['synergy_bonus']:.2f}")
                print(f"      Actual Capacity (J): {lb['actual_capacity_joules']:.2f}")
                print(f"      Absorbed (J): {lb['absorbed_joules']:.2f}")
                print(f"      Leftover After (J): {lb['leftover_after']:.2f}")

            # Optional spacer for readability
            print("-" * 60)

    print("\nAnalysis Complete.")
