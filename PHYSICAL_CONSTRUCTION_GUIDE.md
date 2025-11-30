# Paracletic Warp Coil: Physical Construction Guide

## ⚠️ CRITICAL DISCLAIMER

This simulation uses **abstract "Paracletic" mathematics** — NOT Maxwell's electromagnetic equations. The field shapes shown are conceptual visualizations, not physical EM field predictions.

To build a real coil that produces actual EM fields matching these patterns, you would need to:
1. Use the geometry parameters from this tool as a starting point
2. Validate with actual EM simulation software (COMSOL, ANSYS, etc.)
3. Prototype and measure with proper RF equipment

---

## How To Run The Simulation

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone/navigate to project
cd paracletic_warp_coil

# Run
cargo run --release
```

### Controls Summary

| Key | Action |
|-----|--------|
| **Space** | Pause/resume time |
| **C** | Switch between OUTER and INNER coil editing |
| **1-8** | Select Paraclete parameter |
| **9,0,-,=** | Select shell parameter |
| **↑↓** | Fine adjust selected parameter |
| **←→** | Coarse adjust selected parameter |
| **G** | Toggle contour view (shows field strength levels) |
| **V** | Toggle vector field display |
| **B** | Toggle emission field display |
| **X** | Toggle cross-section graph |
| **Z/.** | Rotate cross-section angle |
| **S** | Toggle physical specs display |
| **M** | Toggle metrics display |
| **I/O** | Toggle inner/outer coil visibility |
| **R** | Reset all parameters to defaults |

---

## Reading The Physical Specifications

When you press **S** to show physical specs, you'll see:

```
═══ PHYSICAL COIL SPECS ═══
Major Radius: 15.0 cm       ← R: center of torus to tube center
Minor Radius: 2.48 cm       ← r: tube/winding radius  
Total Turns: 144 (12/sector) ← Wire wraps around the tube
Sectors: 12                  ← Logical divisions for phased drive
Wire: 18 AWG (1.02 mm)      ← Wire thickness
Wire Length: 2.5 m          ← Total wire needed
Copper Mass: 23 g           ← Weight of copper
Inductance: 4.2 µH          ← Calculated from geometry
Self-Resonance: 24.5 MHz    ← With ~10pF parasitic capacitance
Root Freq: 25000 Hz         ← Target harmonic (root channels)
Crown Freq: 33000 Hz        ← Target harmonic (crown channels)
```

### How Parameters Map to Physical Specs

| Paraclete Param | Physical Effect |
|-----------------|-----------------|
| `root_freq` | Root harmonic frequency = base_freq × root_freq × 100 |
| `crown_freq` | Crown harmonic frequency = base_freq × crown_freq × 100 |
| `base_amp` | Affects minor radius (higher = thicker coil) |
| `drive_amp` | Affects total turns (higher = more turns) |

---

## Understanding The Field Shape

### What You're Seeing

The visualization shows a **2D cross-section** of a toroidal (donut-shaped) coil:

```
        ┌─────────────────┐
       /                   \
      │    Inner Bubble     │  ← Inner coil (if enabled)
      │   ┌───────────┐    │
      │   │   NULL    │    │  ← Quiet zone at center
      │   │   ZONE    │    │
      │   └───────────┘    │
      │                     │
       \                   /
        └─────────────────┘
              ↑
        Outer Coil (Warp Lattice)
```

### Field Displays

1. **Emission View (B key)**: Shows energy distribution
   - Brighter = more energy
   - White = high coherence/resonance
   - Colors indicate harmonic balance

2. **Contour View (G key)**: Shows field magnitude levels
   - Blue → Cyan → Green → Yellow → Red (low to high)
   - White contour lines show iso-field boundaries
   - **This is the "shape" you want for construction**

3. **Vector View (V key)**: Shows field direction
   - Arrows point in field direction
   - Longer/brighter = stronger field
   - Look for circulation patterns

4. **Cross-Section (X key)**: Shows 1D slice through the field
   - Green line = field magnitude along the slice
   - Orange vertical lines = outer coil position
   - Blue vertical lines = inner coil position
   - **Use this to see the field profile**

---

## Translating To Physical Construction

### Step 1: Determine Your Target Frequency

The simulation uses a **base frequency** of 1000 Hz by default. Your actual harmonic frequencies are:

```
Root Frequency = base_freq × root_freq × 100
Crown Frequency = base_freq × crown_freq × 100

Example with defaults:
Root = 1000 × 0.25 × 100 = 25,000 Hz (25 kHz)
Crown = 1000 × 0.33 × 100 = 33,000 Hz (33 kHz)
```

**To change base frequency**: Edit `ui.base_freq_hz` in the code, or adjust `root_freq`/`crown_freq` params to scale.

### Step 2: Get Physical Dimensions

From the PHYSICAL COIL SPECS panel:

```
OUTER COIL (switch with C key):
┌────────────────────────────────────┐
│ Major Radius (R):  ____ cm         │  ← Measure from center
│ Minor Radius (r):  ____ cm         │  ← Tube diameter ÷ 2
│ Total Turns:       ____            │  ← Winds around tube
│ Turns per Sector:  ____            │  ← For phased drive
│ Sectors:           12              │  ← Angular divisions
│ Wire Gauge:        ____ AWG        │  ← Wire size
│ Wire Length:       ____ m          │  ← How much to buy
└────────────────────────────────────┘

INNER COIL (switch with C key):
┌────────────────────────────────────┐
│ (Same format, smaller dimensions)   │
│ Radius = Outer × inner_radius_ratio │
└────────────────────────────────────┘
```

### Step 3: Understand The 12-Sector Layout

The coil is divided into **12 sectors** around the torus. Each sector can be:

1. **Wound as one continuous coil** (simple construction)
2. **Wound separately and phased** (advanced: allows harmonic control)

```
        Sector 0 (0°)
            │
    11 ─────┼───── 1
           / \
   10 ────/   \──── 2
         │     │
    9 ───│     │─── 3
         │     │
    8 ────\   /──── 4
           \ /
     7 ─────┼───── 5
            │
        Sector 6 (180°)
```

**For phased drive**: Each sector gets its signal phase-shifted by `30° × sector_number`.

### Step 4: Construct The Form

**Toroidal Form Options:**

1. **3D Printed**: Print a hollow torus, wind wire around it
2. **Ferrite Core**: Buy toroidal ferrite, wind directly
3. **Air Core**: Use a flexible tube bent into a circle

**Dimensions from simulation** (example with 15cm major radius):

```
Outer Diameter: 2 × (R + r) = 2 × (15 + 2.5) = 35 cm
Inner Diameter: 2 × (R - r) = 2 × (15 - 2.5) = 25 cm
Tube Diameter:  2 × r = 5 cm
```

### Step 5: Wind The Coil

```
1. Mark 12 sector boundaries on your form (every 30°)

2. For each sector:
   - Wind (total_turns / 12) turns tightly
   - Keep track of wire direction (all same, or alternating)
   
3. Connection options:
   a) Series: Connect end of sector N to start of sector N+1
   b) Phased: Bring out taps at each sector for external drive
```

---

## Driving The Coil With Harmonics

### Simple Approach (Single Frequency)

Drive the entire coil with a single signal at the self-resonant frequency shown in the specs.

### Advanced Approach (Phased Harmonics)

To replicate the simulation's "root" and "crown" channels:

```
ROOT CHANNELS (sectors 0-5, indices 0-5 in ParState):
- Drive at root_freq_hz
- Phase each sector: φ = sector_angle × phase_speed

CROWN CHANNELS (sectors 6-11, indices 6-11 in ParState):
- Drive at crown_freq_hz  
- Phase each sector: φ = sector_angle × phase_speed + 90°
```

**Hardware needed:**
- Multi-channel signal generator or FPGA
- 12 amplifier channels (or 2 with multiplexing)
- Phase control per channel

---

## What The Simulation CANNOT Tell You

❌ **Actual EM field shape** — The Paraclete math is not physics  
❌ **RF interference patterns** — No wave equation solved  
❌ **Impedance matching** — Depends on your driver circuit  
❌ **Thermal behavior** — No heat modeling  
❌ **Material effects** — No core permeability modeled  

✅ **What it CAN give you:**
- Starting geometry (radii, turns, sectors)
- Target frequencies and their ratios
- Visual intuition for the field "shape" you're aiming for
- A parameter space to explore

---

## Recommended Next Steps

1. **Export your parameters** — Write down all specs from the panel

2. **Model in EM software** — Take the geometry into COMSOL/ANSYS/FEMM:
   ```
   - Draw toroidal coil with your dimensions
   - Set conductor as copper, your wire gauge
   - Apply your target frequencies
   - Solve for actual B-field distribution
   ```

3. **Build a prototype** — Start with the outer coil only:
   ```
   - 3D print or source a toroidal form
   - Wind with specified turns and wire
   - Measure actual inductance with LCR meter
   - Compare to simulation's calculated value
   ```

4. **Test with simple drive** — Before phased harmonics:
   ```
   - Drive at self-resonance
   - Measure field with pickup coil
   - Compare field shape to simulation
   ```

5. **Iterate** — The simulation helps you explore, but reality requires tuning.

---

## Example Build Specification

Based on default parameters:

```
OUTER WARP LATTICE COIL
═══════════════════════════════════════
Form Factor:        Toroid
Major Radius (R):   15.0 cm
Minor Radius (r):   2.48 cm
Total Turns:        144
Sectors:            12
Turns/Sector:       12
Wire:               18 AWG copper
Wire Length:        ~2.5 m
Inductance:         ~4.2 µH (calculated)

Target Frequencies:
  Root:             25,000 Hz
  Crown:            33,000 Hz
  Self-Resonance:   ~24.5 MHz

INNER BUBBLE COIL (if building both)
═══════════════════════════════════════
Radius:             6.75 cm (45% of outer)
Turns:              ~86 (60% of outer)
Sectors:            8
Phase Offset:       0.0 (adjustable)
```

---

## Field Shape Summary

The target field configuration:

```
┌──────────────────────────────────────────┐
│                                          │
│      ████ HIGH FIELD (coil region) ████  │
│    ██                              ██    │
│   █    ┌──────────────────────┐     █   │
│   █    │                      │     █   │
│  █     │   LOW FIELD (NULL)   │      █  │
│  █     │                      │      █  │
│   █    │      (flat, quiet)   │     █   │
│   █    └──────────────────────┘     █   │
│    ██                              ██    │
│      ████████████████████████████████      │
│                                          │
└──────────────────────────────────────────┘

Cross-section profile:
         │
Field    │    ╱╲        ╱╲
Strength │   ╱  ╲      ╱  ╲
         │  ╱    ╲____╱    ╲
         │ ╱                 ╲
         └────────────────────────
           outer   null   outer
           coil    zone   coil
```

The goal is:
- **High field at coil location** (the "shell")
- **Low, flat field in the center** (the "bubble")
- **Sharp transition** controllable via parameters

Press **X** in the simulation to see this cross-section profile in real-time.
