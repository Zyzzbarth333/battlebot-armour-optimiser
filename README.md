# BattleBot Armour Optimisation System

## Overview

This Python module provides a **multi-layer armour** optimisation tool for combat robots (e.g., “BattleBots”). It calculates how different material stacks (up to 5 layers) handle three types of high-energy impacts (hammer, spinner, and saw), while respecting:

- **Maximum total weight** for armour
- **Overall thickness limits**
- **Cost constraints** (in relative units)

Using a **bilinear stress–strain** approximation with optional hardness-based scaling, the system estimates how much energy each layer absorbs. The goal is to find the *best* layer order and thicknesses to **fully absorb** the weapon’s impact energy (protection score = 1.00) without exceeding weight limits.

---

## Features & Highlights

1. **Layer-by-Layer Calculations**  
   - Each layer sees the **remaining** energy from the previous layer.  
   - Once a layer’s capacity is “saturated,” leftover energy passes on to the next layer.

2. **Advanced (Bilinear) Absorption Model**  
   - For each material, the system computes energy absorption from yield strength, ultimate strength, and elastic modulus.  
   - Incorporates partial **hardness scaling** for spinner and saw impacts.

3. **Handles Up to 5 Layers**  
   - Permutations of 1–5 layers are generated, and each is tested for weight & thickness feasibility.

4. **Impact Energies**  
   - Hammer: 15,000 J  
   - Spinner: 18,000 J  
   - Saw: 13,000 J

5. **Weight Constraints**  
   - Example usage: total robot mass = 13.61 kg, with 60% reserved for internals ⇒ 5.4 kg remaining for armour.

6. **Cost and Sorting**  
   - Results are sorted primarily by **protection score** (absorbed energy / total incoming).  
   - Ties at 100% absorption can then be examined for weight, cost, or thickness.

---

## Material Database (Updated)

Below are the **updated** typical properties for battlebot use:

| Material              | Density (kg/m³) | Yield (MPa) | Ultimate (MPa) | Elastic Modulus (GPa) | Hardness (HB) | Cost (rel. units/kg) |
|-----------------------|-----------------|-------------|----------------|-----------------------|---------------|----------------------|
| **HDPE**             | 960             | 26          | 33             | 1.0                   | 60            | 2.5                  |
| **TPU**              | 1200            | 25          | 40             | 0.05                  | 80            | 3.0                  |
| **Aluminium 6061**   | 2700            | 275         | 310            | 69                    | 95            | 4.0                  |
| **Steel 4140**       | 7850            | 900         | 1080           | 210                   | 300           | 5.0                  |
| **Titanium Ti-6Al-4V** | 4430          | 950         | 1050           | 113                   | 340           | 8.0                  |

**Notes**:  
- These values are representative “mid-range” data for common battlebot alloys and processing. Actual properties vary by heat treatment.  
- Cost is relative and used for comparative optimisation (not an absolute currency).

---

## How It Works

1. **Generate Valid Configurations**  
   - For `n` layers (1 to `max_layers`), the system attempts all permutations of materials.  
   - Each layer’s thickness is chosen from a discrete range (e.g., 0.5 to 6.0 mm in 0.5 mm steps) so long as the **total thickness** does not exceed `max_thickness`.  
   - Any config that surpasses the **armour weight limit** is discarded.

2. **Evaluate Armour Config**  
   For each valid config:
   - **Initial** impact energy: 15k / 18k / 13k J (hammer/spinner/saw).  
   - **Layer Absorption**:  
     1. Convert thickness to metres.  
     2. Compute **stress–strain** area (elastic + plastic) → J/m³.  
     3. Scale by thickness, area, and (for spinner/saw) possibly by **hardness**.  
     4. Each layer absorbs up to its capacity or the remaining energy, whichever is smaller.  
   - Subtract absorbed energy from “remaining.” If leftover > 0, pass it to the next layer.  
   - Summarise total weight, cost, and final **protection score** = (energy absorbed / initial energy).

3. **Sort & Output**  
   - The best configurations (up to top 5) are displayed for each impact type.  
   - Each result shows how much each layer absorbed, final leftover energy (often zero if fully absorbed), total weight, and cost.

---

## Constraints & Assumptions

1. **Maximum Armour Weight**  
   - E.g., if your total robot mass is 13.61 kg, you might reserve 8.2 kg (60%) for internals, leaving **5.4 kg** for armour.

2. **Maximum Total Thickness**  
   - Default is **6.0 mm** combined across all layers.

3. **Discrete Thickness Steps**  
   - Typical increments: **0.5 mm** or **1.0 mm** in the `_generate_thicknesses` function.

4. **Uniform Impact and Area**  
   - The system calculates total absorption capacity by multiplying J/m² by total external surface area.  
   - Real local hits, seam weaknesses, etc. are not simulated.

5. **No Deflection or Puncture Criteria**  
   - The model purely integrates a bilinear stress–strain curve. It does not consider large deflections, local tearing, or strain-rate effects.

6. **Cost Is Simplistic**  
   - Cost is purely mass × cost_per_kg. No machining or shaping complexities are included.

7. **No Synergy or Interface Effects**  
   - Layers do not degrade each other or combine in advanced ways (like adhesives, stress concentrations at layer boundaries, etc.).

8. **Saw/Spinner Hardness Scaling**  
   - For **spinner**: hardness factor scales the bilinear area to reflect indentation resistance.  
   - For **saw**: partial friction factor also uses hardness. These are approximate heuristics.

---

## Example Console Output (Excerpts)

Below are sample “top 5” outputs for **hammer** (15,000 J) and **spinner** (18,000 J):

```
============================================================
TOP 5 FOR HAMMER IMPACTS
============================================================

Rank 1:
Total Thickness: 5.0 mm
Outer Layer: TPU (3.0 mm)
Middle Layer: Steel (1.0 mm)
Inner Layer: Titanium (1.0 mm)
Protection Score: 1.00
Total Absorbed Energy: 15000.0 J
Total Weight: 4.29 kg
Total Cost: 23.08 units

Layer-by-Layer Breakdown:
  Layer 0: TPU
    Thickness: 3.0 mm
    Absorption Capacity (total): 12960.0 J
    Incoming Energy: 15000.0 J
    Energy Absorbed: 12960.0 J
    Energy Remaining: 2040.0 J
  Layer 1: Steel
    Thickness: 1.0 mm
    Absorption Capacity (total): 749.8 J
    Incoming Energy: 2040.0 J
    Energy Absorbed: 749.8 J
    Energy Remaining: 1290.2 J
  Layer 2: Titanium
    Thickness: 1.0 mm
    Absorption Capacity (total): 1317.1 J
    Incoming Energy: 1290.2 J
    Energy Absorbed: 1290.2 J
    Energy Remaining: 0.0 J
------------------------------------------------------------

...
```

And for **spinner** (18,000 J):
```
============================================================
TOP 5 FOR SPINNER IMPACTS
============================================================

Rank 1:
Total Thickness: 5.5 mm
Outer Layer: TPU (3.0 mm)
Middle Layer: Steel (1.5 mm)
Inner Layer: Titanium (1.0 mm)
Protection Score: 1.00
Total Absorbed Energy: 18000.0 J
Total Weight: 5.35 kg
Total Cost: 28.38 units

Layer-by-Layer Breakdown:
  Layer 0: TPU
    Thickness: 3.0 mm
    Absorption Capacity (total): 10368.0 J
    Incoming Energy: 18000.0 J
    Energy Absorbed: 10368.0 J
    Energy Remaining: 7632.0 J
  Layer 1: Steel
    Thickness: 1.5 mm
    Absorption Capacity (total): 3374.2 J
    Incoming Energy: 7632.0 J
    Energy Absorbed: 3374.2 J
    Energy Remaining: 4257.8 J
  Layer 2: Titanium
    Thickness: 1.0 mm
    Absorption Capacity (total): 4478.3 J
    Incoming Energy: 4257.8 J
    Energy Absorbed: 4257.8 J
    Energy Remaining: 0.0 J
------------------------------------------------------------
```

---

## Getting Started

1. **Clone or Download**  
   ```bash
   git clone <repository_url>
   cd battlebot-armour-optimiser
   ```
2. **Install Requirements**  
   ```bash
   pip install -r requirements.txt
   ```
   - Python 3.7+ recommended (uses `dataclasses` & `numpy`).

3. **Run the Main Script**  
   ```bash
   python ConstrainedArmourTest.py
   ```
   - Check the console for top-5 results for **hammer**, **spinner**, and **saw**.

4. **Edit Material Properties / Impact Energies**  
   - Open `ConstrainedArmourTest.py`.  
   - Look for `self.materials = { ... }` and `self.impact_energies = { ... }` to tweak values.

---

## Using It in Your Own Code

If you want to integrate this engine into another script:

```python
from ConstrainedArmourTest import ArmorTester, BotSurfaceAreas

# Define your robot's surface areas (m²)
surface_areas = BotSurfaceAreas(
    left=0.045,
    right=0.045,
    front=0.030,
    back=0.030,
    top=0.060,
    bottom=0.060
)

# Create an instance with your max weight, layering, etc.
tester = ArmorTester(
    surface_areas=surface_areas,
    max_total_weight=13.61,  # e.g., 13.61 kg total
    max_layers=5,
    max_thickness=6.0
)

# Evaluate top 5 configurations for spinner
spinner_results = tester.test_armor_configurations('spinner')

# Print or process results
for rank, result in enumerate(spinner_results, 1):
    print(f"Rank {rank}: {result}")
```

---

## Future Directions & Caveats

1. **More Realistic Fracture & Deflection**  
   - Currently, the code uses a bilinear stress–strain model and does not account for large deflections or potential tearing in polymers.  
2. **Strain-Rate Effects**  
   - Hammer, spinner, or saw impacts often occur at high strain rates. Actual metal or polymer performance could differ.  
3. **Manufacturing Feasibility**  
   - No penalty for complex shapes or expensive processes.  
4. **Multi-Objective Optimisation**  
   - At present, we primarily rank by **protection_score**. You could incorporate cost or mass into a combined metric if desired.  
5. **Local vs. Global Impacts**  
   - Uniform area assumption might underestimate local penetrations at edges, corners, or bolt holes.

---

## Licence

This project is open-source—feel free to adapt or extend it for your own combat robot experiments. See the repository’s LICENCE file for details.

---
