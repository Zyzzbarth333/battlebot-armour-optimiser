from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from itertools import combinations, permutations


@dataclass
class Material:
    """
    Encapsulates material properties in SI units for use in armour analysis.

    Attributes:
        name (str): Material name (e.g. 'Steel 4140').
        density (float): Density in kg/m³.
        yield_strength (float): Yield strength in MPa.
        ultimate_strength (float): Ultimate tensile strength in MPa.
        elastic_modulus (float): Elastic modulus in GPa.
        hardness (float): Brinell hardness number (dimensionless).
        cost_per_kg (float): Relative cost per kg (unitless).
    """
    name: str
    density: float  # kg/m³
    yield_strength: float  # MPa
    ultimate_strength: float  # MPa
    elastic_modulus: float  # GPa
    hardness: float  # Brinell hardness
    cost_per_kg: float  # Relative cost units

    def get_weight_per_area(self, thickness_mm: float) -> float:
        """
        Calculate weight per square metre (kg/m²) for a given thickness in millimetres.

        Args:
            thickness_mm (float): Layer thickness in mm.

        Returns:
            float: Weight in kg per square metre of the material layer.
        """
        return self.density * (thickness_mm / 1000.0)


@dataclass
class BotSurfaceAreas:
    """
    Holds the surface areas (m²) of each face of the robot.

    Attributes:
        left (float): Left face area in m².
        right (float): Right face area in m².
        front (float): Front face area in m².
        back (float): Back face area in m².
        top (float): Top face area in m².
        bottom (float): Bottom face area in m².
    """
    left: float
    right: float
    front: float
    back: float
    top: float
    bottom: float

    def get_surface_areas(self) -> Dict[str, float]:
        """
        Return a dictionary of surface areas keyed by face name.

        Returns:
            Dict[str, float]: Dictionary mapping face names to area in m².
        """
        return {
            'left': self.left,
            'right': self.right,
            'front': self.front,
            'back': self.back,
            'top': self.top,
            'bottom': self.bottom
        }


class ArmourTester:
    """
    Main class for generating and evaluating multi-layer armour configurations.

    It can handle up to `max_layers` layers, each with varying thickness,
    and checks feasibility based on total weight and thickness constraints.
    """

    def __init__(self,
                 surface_areas: BotSurfaceAreas,
                 max_total_weight: float,
                 max_layers: int = 3,
                 max_thickness: float = 6.0):
        """
        Initialise the armour testing system.

        Args:
            surface_areas (BotSurfaceAreas): The robot's face areas in m².
            max_total_weight (float): Total robot weight limit in kg.
            max_layers (int): Maximum number of layers the armour can have.
            max_thickness (float): Maximum total armour thickness in mm.
        """
        # Store the robot's surface areas in a dictionary
        self.surface_areas = surface_areas.get_surface_areas()

        # Store input parameters
        self.max_total_weight = max_total_weight
        self.max_layers = max_layers
        self.max_thickness = max_thickness

        # Reserve 60% of total weight for internals, so only the remainder is available for armour
        self.base_weight = max_total_weight * 0.6
        self.armour_weight_limit = max_total_weight - self.base_weight

        # Common materials with approximate mechanical properties
        self.materials = {
            'HDPE': Material(
                name='HDPE',
                density=960,            # kg/m³
                yield_strength=26,      # MPa
                ultimate_strength=33,   # MPa
                elastic_modulus=1.0,    # GPa
                hardness=60,            # Brinell
                cost_per_kg=2.5
            ),
            'TPU': Material(
                name='TPU',
                density=1200,           # kg/m³
                yield_strength=25,      # MPa
                ultimate_strength=40,   # MPa
                elastic_modulus=0.05,   # GPa
                hardness=80,            # Brinell
                cost_per_kg=3.0
            ),
            'Aluminium': Material(
                name='Aluminium 6061',
                density=2700,           # kg/m³
                yield_strength=275,     # MPa (6061-T6 typical)
                ultimate_strength=310,  # MPa
                elastic_modulus=69,     # GPa
                hardness=95,            # Brinell
                cost_per_kg=4.0         # Relative cost scale
            ),
            'Steel': Material(
                name='Steel 4140',
                density=7850,           # kg/m³
                yield_strength=900,     # MPa (Q&T for battlebot usage)
                ultimate_strength=1080, # MPa
                elastic_modulus=210,    # GPa
                hardness=300,           # ~300 Brinell for a moderate Q&T
                cost_per_kg=5.0
            ),
            'Titanium': Material(
                name='Titanium Ti-6Al-4V',
                density=4430,           # kg/m³
                yield_strength=950,     # MPa
                ultimate_strength=1050, # MPa
                elastic_modulus=113,    # GPa
                hardness=340,           # Brinell
                cost_per_kg=8.0
            )
        }

        # Approximate impact energy levels in Joules for different weapon types
        self.impact_energies = {
            'hammer': 15000.0,
            'spinner': 18000.0,
            'saw': 13000.0
        }

    def advanced_energy_absorption(self,
                                   material: Material,
                                   thickness_mm: float,
                                   impact_type: str) -> float:
        """
        Compute a more realistic energy absorption capacity (J/m²) using a bilinear
        stress–strain approach (elastic + plastic).

        For 'spinner' and 'saw' impacts, a hardness-based scaling factor is also applied.

        Args:
            material (Material): Material object with mechanical properties.
            thickness_mm (float): Layer thickness in millimetres.
            impact_type (str): Type of impact ('hammer', 'spinner', or 'saw').

        Returns:
            float: Energy absorption capacity per square metre (J/m²).
        """
        # Convert thickness to metres
        t_m = thickness_mm / 1000.0

        # Convert material strengths/modulus to Pascals
        sigma_y = material.yield_strength * 1e6     # Pa
        sigma_u = material.ultimate_strength * 1e6  # Pa
        E_mod = material.elastic_modulus * 1e9      # Pa

        # Calculate strain at yield and ultimate
        eps_y = sigma_y / E_mod
        eps_u = sigma_u / E_mod

        # Area under the elastic region (triangle under stress–strain up to yield)
        elastic_energy = 0.5 * sigma_y * eps_y  # J/m³

        # Approximate plastic region up to ultimate (linear from yield to ultimate)
        if eps_u > eps_y:
            plastic_energy = 0.5 * (sigma_y + sigma_u) * (eps_u - eps_y)  # J/m³
        else:
            plastic_energy = 0.0  # If ultimate <= yield, no plastic region

        # Total energy density in J/m³
        total_energy_density = elastic_energy + plastic_energy

        # Scale by thickness to get J/m²
        energy_absorption_per_m2 = total_energy_density * t_m

        # Optional impact-specific scaling
        if impact_type == 'spinner':
            # Hardness scaling
            reference_hardness = 100.0
            hardness_factor = material.hardness / reference_hardness
            energy_absorption_per_m2 *= hardness_factor

        elif impact_type == 'saw':
            # Simple friction factor placeholder
            friction_factor = (material.hardness / 200.0)
            energy_absorption_per_m2 *= (1 + 0.3 * friction_factor)

        # Hammer case uses the direct bilinear model without extra scaling
        return energy_absorption_per_m2

    def get_valid_thickness_range(self, material_name: str) -> Tuple[float, float, float]:
        """
        Specify the allowable thickness range (mm) for each material,
        based on engineering judgements or limitations.

        Args:
            material_name (str): Name of the material (e.g., 'Steel').

        Returns:
            (float, float, float): (min_thickness, max_thickness, step) in mm.
        """
        if material_name in ['HDPE', 'TPU']:
            return (0.5, min(3.0, self.max_thickness), 0.5)
        elif material_name == 'Aluminium':
            return (0.5, min(2.0, self.max_thickness), 0.5)
        elif material_name == 'Steel':
            return (0.5, min(2.0, self.max_thickness), 0.5)
        elif material_name == 'Titanium':
            return (0.5, min(1.0, self.max_thickness), 0.5)
        return (0.5, min(6.0, self.max_thickness), 0.5)

    def calculate_config_weight(self,
                                materials: List[str],
                                thicknesses: List[float]) -> float:
        """
        Calculate the total weight (kg) of an armour configuration across all faces.

        Args:
            materials (List[str]): Names of the materials in each layer.
            thicknesses (List[float]): Corresponding thicknesses for each layer in mm.

        Returns:
            float: Total mass (kg) of the entire armour, summing over all faces.
        """
        total_weight = 0.0
        for side_area in self.surface_areas.values():
            for material_name, thickness in zip(materials, thicknesses):
                mat = self.materials[material_name]
                layer_mass = mat.get_weight_per_area(thickness) * side_area
                total_weight += layer_mass
        return total_weight

    def evaluate_armour_config(self,
                               materials: List[str],
                               thicknesses: List[float],
                               impact_type: str) -> Dict:
        """
        Evaluate the effectiveness of a specific multi-layer armour configuration
        against a given impact type.

        Each layer receives the full 'remaining' energy from the previous layer;
        it either fully or partially absorbs the energy, passing any leftover on.

        Args:
            materials (List[str]): Material names for each layer (outermost first).
            thicknesses (List[float]): Thickness (mm) for each corresponding layer.
            impact_type (str): One of {'hammer', 'spinner', 'saw'}.

        Returns:
            Dict: Dictionary containing the protection score, total weight, total cost,
                  total absorbed energy, remaining energy, and layer-by-layer breakdown.
        """
        # Initial energy for this impact type
        impact_energy = self.impact_energies[impact_type]
        current_energy = impact_energy

        # Calculate total weight & cost as we sum across layers
        total_weight = self.calculate_config_weight(materials, thicknesses)
        total_cost = 0.0

        # Record the absorption details per layer
        layer_results = []

        # Evaluate each layer in sequence
        for i, (material_name, thickness) in enumerate(zip(materials, thicknesses)):
            mat = self.materials[material_name]

            # Calculate layer cost
            weight_per_m2 = mat.get_weight_per_area(thickness)
            layer_cost = sum(
                weight_per_m2 * area * mat.cost_per_kg
                for area in self.surface_areas.values()
            )
            total_cost += layer_cost

            # Use the bilinear stress–strain approach
            absorption_per_m2 = self.advanced_energy_absorption(mat, thickness, impact_type)

            # Multiply by total area to get total absorption (J) for this layer
            total_area = sum(self.surface_areas.values())
            absorption_capacity_total = absorption_per_m2 * total_area

            # This layer either absorbs all its capacity or whatever is left
            energy_absorbed = min(absorption_capacity_total, current_energy)
            current_energy -= energy_absorbed

            # Store per-layer results
            layer_results.append({
                'layer_index': i,
                'material': material_name,
                'thickness_mm': thickness,
                'absorption_capacity_total': absorption_capacity_total,
                'incoming_energy': current_energy + energy_absorbed,
                'energy_absorbed': energy_absorbed,
                'energy_remaining': current_energy
            })

            # If everything is absorbed, we can stop processing further layers
            if current_energy <= 0:
                break

        # Calculate final overall performance metrics
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
        """
        Generate all valid armour configurations for a given number of layers,
        respecting the armour weight limit and total thickness constraint.

        Args:
            num_layers (int): Number of layers in each configuration.

        Returns:
            List[Dict]: List of valid configurations, each with materials + thicknesses.
        """
        valid_configs = []
        material_names = list(self.materials.keys())

        # All combinations of materials (e.g., 'Steel', 'TPU') for the required num_layers
        for material_combo in combinations(material_names, num_layers):
            # Permute the chosen combination (i.e. different orderings)
            for combo in permutations(material_combo):
                # Get thickness ranges for each chosen material
                thickness_ranges = [self.get_valid_thickness_range(m) for m in combo]

                # Generate all possible thickness combos within the constraints
                for thicknesses in self._generate_thicknesses(thickness_ranges):
                    total_weight = self.calculate_config_weight(combo, thicknesses)
                    if total_weight <= self.armour_weight_limit:
                        valid_configs.append({
                            'materials': list(combo),
                            'thicknesses': thicknesses
                        })
        return valid_configs

    def _generate_thicknesses(self,
                              thickness_ranges: List[Tuple[float, float, float]]) -> List[List[float]]:
        """
        Recursively generate all valid thickness combinations given a list
        of (min_thickness, max_thickness, step_size) per material.

        Args:
            thickness_ranges (List[Tuple[float, float, float]]):
                Each tuple is (min_t, max_t, step) for one material layer.

        Returns:
            List[List[float]]: A list of lists, where each sub-list is a set
            of thicknesses that sum up to <= self.max_thickness.
        """

        def recursive_thickness_gen(current_th: List[float],
                                    remaining_ranges: List[Tuple[float, float, float]],
                                    remaining_thick: float) -> List[List[float]]:
            """Helper function to build thickness combos layer by layer."""
            if not remaining_ranges:
                return [current_th] if current_th else []

            min_t, max_t, step = remaining_ranges[0]
            all_combos = []
            adjusted_max = min(max_t, remaining_thick)

            # Try all thickness increments in [min_t, adjusted_max]
            for thickness in np.arange(min_t, adjusted_max + step, step):
                if thickness <= remaining_thick:
                    new_th = current_th + [thickness]
                    all_combos.extend(
                        recursive_thickness_gen(
                            new_th,
                            remaining_ranges[1:],
                            remaining_thick - thickness
                        )
                    )
            return all_combos

        return recursive_thickness_gen([], thickness_ranges, self.max_thickness)

    def test_armour_configurations(self, impact_type: str) -> List[Dict]:
        """
        Generate and evaluate all valid armour configurations for the specified impact type,
        for up to 'max_layers'.

        Args:
            impact_type (str): One of {'hammer', 'spinner', 'saw'}.

        Returns:
            List[Dict]: Top 5 configurations, sorted by descending protection score.
        """
        results = []
        # Evaluate from 1-layer up to max_layers
        for num_layers in range(1, self.max_layers + 1):
            print(f"\nGenerating {num_layers}-layer configurations...")
            configs = self.generate_valid_configs(num_layers)
            print(f"Found {len(configs)} valid configurations")

            # Evaluate each configuration
            for config in configs:
                effectiveness = self.evaluate_armour_config(
                    config['materials'],
                    config['thicknesses'],
                    impact_type
                )
                results.append({
                    'materials': config['materials'],
                    'thicknesses': config['thicknesses'],
                    'effectiveness': effectiveness
                })

        # Sort the results by highest protection_score and return top 5
        sorted_results = sorted(
            results,
            key=lambda x: x['effectiveness']['protection_score'],
            reverse=True
        )
        return sorted_results[:5]


def print_results(results: List[Dict], impact_type: str) -> None:
    """
    Print the top results in a formatted, human-readable way.

    Args:
        results (List[Dict]): List of configuration results from `test_armour_configurations`.
        impact_type (str): The type of impact (e.g. 'hammer').
    """
    print(f"\n{'=' * 60}")
    print(f"TOP 5 FOR {impact_type.upper()} IMPACTS")
    print(f"{'=' * 60}")

    for rank, config in enumerate(results, 1):
        print(f"\nRank {rank}:")
        # Summation of thickness (mm) in the chosen config
        total_thickness = sum(config['thicknesses'])
        print(f"Total Thickness: {total_thickness:.1f} mm")

        # Print each layer with a label
        for i, (material, thickness) in enumerate(zip(config['materials'], config['thicknesses'])):
            if i == 0:
                layer_type = "Outer"
            elif i == len(config['materials']) - 1:
                layer_type = "Inner"
            else:
                layer_type = "Middle"
            print(f"{layer_type} Layer: {material} ({thickness:.1f} mm)")

        eff = config['effectiveness']
        print(f"Protection Score: {eff['protection_score']:.2f}")
        print(f"Total Absorbed Energy: {eff['total_absorbed']:.1f} J")
        print(f"Total Weight: {eff['total_weight']:.2f} kg")
        print(f"Total Cost: {eff['total_cost']:.2f} units")

        # Print per-layer breakdown
        print("\nLayer-by-Layer Breakdown:")
        for layer in eff['layer_breakdown']:
            print(f"  Layer {layer['layer_index']}: {layer['material']}")
            print(f"    Thickness: {layer['thickness_mm']:.1f} mm")
            print(f"    Absorption Capacity (total): {layer['absorption_capacity_total']:.1f} J")
            print(f"    Incoming Energy: {layer['incoming_energy']:.1f} J")
            print(f"    Energy Absorbed: {layer['energy_absorbed']:.1f} J")
            print(f"    Energy Remaining: {layer['energy_remaining']:.1f} J")

        print("-" * 60)


if __name__ == "__main__":
    # Example usage with explicit surface areas (in m²)
    surface_areas = BotSurfaceAreas(
        left=0.045,
        right=0.045,
        front=0.030,
        back=0.030,
        top=0.060,
        bottom=0.060
    )

    # Create an instance of ArmourTester
    # Example: total robot mass = 13.61 kg, we allow up to 5 layers, max total thickness = 6 mm
    tester = ArmourTester(
        surface_areas=surface_areas,
        max_total_weight=13.61,
        max_layers=5,
        max_thickness=6.0
    )

    print("Running Armour Analysis (Advanced Absorption)")
    print("=============================================")
    print(f"Maximum Total Weight: {tester.max_total_weight:.1f} kg")
    print(f"Base Weight (Internal Components): {tester.base_weight:.1f} kg")
    print(f"Available Weight for Armour: {tester.armour_weight_limit:.1f} kg")

    print("\nSurface Areas (m²):")
    for side, area in tester.surface_areas.items():
        print(f"  {side.capitalize()}: {area:.3f} m²")

    # Evaluate armour for each impact type and print the top 5
    for impact in ['hammer', 'spinner', 'saw']:
        final_results = tester.test_armour_configurations(impact)
        print_results(final_results, impact)
