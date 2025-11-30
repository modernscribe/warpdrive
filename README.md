# Paracletic Warp Bubble Field Visualizer

## Version 0.2.0 - Multi-Coil Extension

This is an extended version of the Paracletic Warp Coil visualizer with multi-coil support, bubble metrics, and temporal recording.

**Important**: This system is a visual and conceptual simulator of a Paracletic coil field — not a real FTL or warp drive implementation.

---

## New Features

### 1. Multi-Coil Support (WarpBubble)

The system now supports two coils:

- **Outer Coil (Warp Lattice)**: The original 12-sector coil with warm colors
- **Inner Coil (Harmonic Cage Bubble)**: A new 8-sector inner coil with cooler, softer colors

The `WarpBubble` struct manages both coils and handles:
- Coupled field sampling between inner and outer coils
- Combined emission calculations
- Synchronized geometry updates

### 2. Bubble Shell Parameters

New `BubbleShellParams` struct with four parameters:

| Parameter | Description | Range |
|-----------|-------------|-------|
| `inner_radius_ratio` | Inner coil radius as fraction of outer | 0.2 - 0.8 |
| `shell_softness` | Blending factor for inner coil contribution | 0.0 - 1.0 |
| `coupling_strength` | Field coupling between coils | 0.0 - 1.0 |
| `phase_offset` | Temporal phase shift for inner coil | -2.0 - 2.0 |

### 3. Bubble Metrics

Real-time ParState-derived metrics displayed in the control panel:

- **Flatness**: Uniformity of field in the null region (higher = more stable)
- **Shear Index**: Gradient magnitude at bubble boundary (lower = smoother transition)
- **Coherence Ratio**: Correlation between inner and outer coil emissions
- **Null Stability**: Quietness of the null region center

### 4. Temporal Recording System

Record and playback parameter changes:

- Ring buffer stores up to 1024 frames
- Records all Paraclete params and shell params
- Playback drives parameters from recorded data

---

## Controls

### Original Controls (Preserved)

| Key | Action |
|-----|--------|
| Space | Pause/resume time |
| Q/E | Decrease/increase time scale |
| Tab | Cycle selected parameter |
| 1-8 | Select Paraclete parameter directly |
| ↑/↓ | Fine adjustment (+/-) |
| ←/→ | Coarse adjustment (+/-) |
| V | Toggle vector field display |
| B | Toggle emission map display |
| F | Toggle resonance focus mode |
| R | Reset all parameters to defaults |

### New Controls

| Key | Action |
|-----|--------|
| C | Switch active coil (Outer ↔ Inner) |
| I | Toggle inner coil visibility |
| O | Toggle outer coil visibility |
| M | Toggle metrics display |
| 9, 0, -, = | Select shell parameters (1-4) |
| [ | Start recording |
| ] | Stop recording |
| P | Start/stop playback |

---

## Code Architecture

### Core Types (Unchanged)

```
ParState = [f32; 12]  // 12-dimensional state vector
paraclete_chain()     // Canonical operator pipeline: Truth → Purity → Law → Love → Wisdom → Life → Glory
```

### New Types

```rust
BubbleShellParams     // Shell configuration (radius ratio, softness, coupling, phase)
BubbleMetrics         // Computed stability indicators
WarpBubble           // Container for outer + inner coils
TemporalRecorder     // Recording/playback system
CoilType             // Enum: Outer | Inner
```

### Module Structure

```
src/main.rs
├── Paracletic Core Types
├── Bubble Shell Parameters (NEW)
├── Bubble Metrics (NEW)
├── Paracletic Operator Chain (UNCHANGED)
├── Warp Coil Structures (EXTENDED with CoilType)
├── Warp Bubble (NEW)
├── Temporal Recording System (NEW)
├── UI State / Control Panel (EXTENDED)
├── Window / Main Loop (EXTENDED)
└── field3d module (STUB for future 3D extension)
```

---

## Building

```bash
cargo build --release
cargo run --release
```

Requires:
- Rust 2021 edition
- macroquad 0.4

---

## Compliance Notes

This extension follows all constraints from the instruction set:

✅ Paraclete operator chain is untouched  
✅ All visual layers derive from ParState  
✅ Original keybindings preserved  
✅ ParacleteParams struct unchanged  
✅ New parameters in separate structs  
✅ Backward compatible (single coil = show outer only)  
✅ 3D hooks prepared in isolated module  
✅ No external physics equations introduced  

---

## Future Extensions

The `field3d` module provides stubs for 3D visualization:

```rust
pub mod field3d {
    pub fn sample_field_3d(coil: &WarpCoil, t: f32, pos: Vec3) -> Vec3
    pub fn sample_emission_3d(coil: &WarpCoil, t: f32, pos: Vec3) -> (f32, f32, f32, f32)
}
```

These preserve the canonical 2D math via projection while allowing 3D rendering.
