use macroquad::prelude::*;
use std::f32::consts::{PI, TAU};

// =============== Paracletic Core Types ===============

pub const N_DIM: usize = 12;

pub type ParState = [f32; N_DIM];

#[derive(Clone, Copy, Debug)]
pub struct ParacleteParams {
    pub root_freq: f32,
    pub crown_freq: f32,
    pub phase_speed: f32,
    pub base_amp: f32,
    pub drive_amp: f32,
    pub light_freq: f32,
    pub resonance_target: f32,
    pub resonance_width: f32,
}

impl Default for ParacleteParams {
    fn default() -> Self {
        Self {
            root_freq: 0.25,
            crown_freq: 0.33,
            phase_speed: 0.5,
            base_amp: 0.6,
            drive_amp: 0.4,
            light_freq: 18.0,
            resonance_target: 1.0,
            resonance_width: 0.15,
        }
    }
}

// =============== PHYSICAL COIL SPECIFICATIONS ===============
// This maps the abstract parameters to real-world construction specs

#[derive(Clone, Copy, Debug)]
pub struct PhysicalCoilSpec {
    // Toroidal geometry
    pub major_radius_cm: f32,      // R - center of torus to center of tube
    pub minor_radius_cm: f32,      // r - radius of the tube/winding area
    pub num_turns: u32,            // Total wire turns
    pub num_sectors: u32,          // Logical divisions (for phased drive)
    pub turns_per_sector: u32,     // Turns in each sector
    
    // Wire properties
    pub wire_gauge_awg: u32,
    pub wire_diameter_mm: f32,
    
    // Derived electromagnetic properties
    pub inductance_uh: f32,        // Microhenries
    pub resonant_freq_hz: f32,     // With estimated capacitance
    
    // Frequency mapping from Paraclete params
    pub root_freq_hz: f32,
    pub crown_freq_hz: f32,
}

impl PhysicalCoilSpec {
    /// Create physical spec from abstract parameters and desired base frequency
    pub fn from_paraclete(params: &ParacleteParams, base_freq_hz: f32, major_radius_cm: f32) -> Self {
        let num_sectors = 12u32;
        
        // Map abstract frequencies to physical harmonics
        // root_freq and crown_freq are multipliers of base frequency
        let root_freq_hz = base_freq_hz * params.root_freq * 100.0;
        let crown_freq_hz = base_freq_hz * params.crown_freq * 100.0;
        
        // Derive geometry from energy parameters
        // Higher base_amp = thicker coil (more copper mass)
        let minor_radius_cm = major_radius_cm * 0.15 * (0.5 + params.base_amp);
        
        // Turns derived from drive_amp and desired inductance
        // More turns = higher inductance = lower resonant freq
        let base_turns = 144u32; // 12 sectors * 12 turns base
        let turns_modifier = 1.0 + (params.drive_amp - 0.4) * 2.0;
        let num_turns = (base_turns as f32 * turns_modifier.max(0.5)) as u32;
        let turns_per_sector = num_turns / num_sectors;
        
        // Wire gauge - thinner wire for higher frequencies
        let wire_gauge_awg = if root_freq_hz > 100000.0 { 24 }
            else if root_freq_hz > 10000.0 { 20 }
            else if root_freq_hz > 1000.0 { 18 }
            else { 16 };
        
        let wire_diameter_mm = awg_to_mm(wire_gauge_awg);
        
        // Calculate inductance (toroidal coil formula)
        // L = μ₀ * N² * r² / (2 * R)  (simplified)
        let mu_0 = 4.0 * PI * 1e-7; // H/m
        let r_m = minor_radius_cm / 100.0;
        let big_r_m = major_radius_cm / 100.0;
        let n = num_turns as f32;
        let inductance_h = mu_0 * n * n * r_m * r_m / (2.0 * big_r_m);
        let inductance_uh = inductance_h * 1e6;
        
        // Estimate resonant frequency with parasitic capacitance (~10pF typical)
        let capacitance_f = 10e-12;
        let resonant_freq_hz = 1.0 / (TAU * (inductance_h * capacitance_f).sqrt());
        
        Self {
            major_radius_cm,
            minor_radius_cm,
            num_turns,
            num_sectors,
            turns_per_sector,
            wire_gauge_awg,
            wire_diameter_mm,
            inductance_uh,
            resonant_freq_hz,
            root_freq_hz,
            crown_freq_hz,
        }
    }
    
    pub fn winding_length_m(&self) -> f32 {
        // Approximate wire length needed
        let circumference = TAU * self.minor_radius_cm / 100.0;
        let turns_length = circumference * self.num_turns as f32;
        // Add ~10% for connections between sectors
        turns_length * 1.1
    }
    
    pub fn total_copper_mass_g(&self) -> f32 {
        let wire_area_mm2 = PI * (self.wire_diameter_mm / 2.0).powi(2);
        let wire_area_m2 = wire_area_mm2 * 1e-6;
        let length_m = self.winding_length_m();
        let volume_m3 = wire_area_m2 * length_m;
        let copper_density_kg_m3 = 8960.0;
        volume_m3 * copper_density_kg_m3 * 1000.0 // grams
    }
}

fn awg_to_mm(awg: u32) -> f32 {
    // AWG to diameter in mm
    0.127 * 92.0_f32.powf((36.0 - awg as f32) / 39.0)
}

// =============== Bubble Shell Parameters ===============

#[derive(Clone, Copy, Debug)]
pub struct BubbleShellParams {
    pub inner_radius_ratio: f32,
    pub shell_softness: f32,
    pub coupling_strength: f32,
    pub phase_offset: f32,
}

impl Default for BubbleShellParams {
    fn default() -> Self {
        Self {
            inner_radius_ratio: 0.45,
            shell_softness: 0.6,
            coupling_strength: 0.3,
            phase_offset: 0.0,
        }
    }
}

// =============== Bubble Metrics ===============

#[derive(Clone, Copy, Debug, Default)]
pub struct BubbleMetrics {
    pub flatness: f32,
    pub shear_index: f32,
    pub coherence_ratio: f32,
    pub null_stability: f32,
}

// =============== Paracletic Operator Chain ===============

fn principle_truth(mut x: ParState, t: f32) -> ParState {
    let w = 0.5 + 0.5 * (t * 0.37).sin();
    for i in 0..N_DIM {
        x[i] = (x[i] * w).tanh();
    }
    x
}

fn principle_purity(mut x: ParState, t: f32) -> ParState {
    let gate = (t * 0.23).cos().abs();
    let mean: f32 = x.iter().sum::<f32>() / (N_DIM as f32);
    for i in 0..N_DIM {
        x[i] = (x[i] - mean) * gate;
    }
    x
}

fn principle_law(mut x: ParState, _t: f32) -> ParState {
    for i in 0..N_DIM {
        let idx_pair = (i + 6) % N_DIM;
        let avg = 0.5 * (x[i] + x[idx_pair]);
        x[i] = avg;
        x[idx_pair] = avg;
    }
    x
}

fn principle_love(mut x: ParState, t: f32) -> ParState {
    let phase = t * 0.41;
    for i in 0..N_DIM {
        let s = (phase + i as f32 * TAU / N_DIM as f32).sin();
        x[i] = x[i] + 0.25 * s;
    }
    x
}

fn principle_wisdom(mut x: ParState, _t: f32) -> ParState {
    let mut acc = 0.0;
    for i in 0..N_DIM {
        acc += x[i] * (i as f32 + 1.0);
    }
    let scale = 1.0 / (1.0 + acc.abs());
    for i in 0..N_DIM {
        x[i] *= scale;
    }
    x
}

fn principle_life(mut x: ParState, t: f32) -> ParState {
    let growth = 1.0 + 0.15 * (t * 0.19).sin();
    for i in 0..N_DIM {
        x[i] = (x[i] * growth).tanh();
    }
    x
}

fn principle_glory(mut x: ParState, t: f32) -> ParState {
    let rot = t * 0.13;
    let cos_r = rot.cos();
    let sin_r = rot.sin();
    for i in 0..6 {
        let j = i + 6;
        let a = x[i];
        let b = x[j];
        x[i] = a * cos_r - b * sin_r;
        x[j] = a * sin_r + b * cos_r;
    }
    x
}

pub fn paraclete_chain(x0: ParState, t: f32) -> ParState {
    let x1 = principle_truth(x0, t);
    let x2 = principle_purity(x1, t);
    let x3 = principle_law(x2, t);
    let x4 = principle_love(x3, t);
    let x5 = principle_wisdom(x4, t);
    let x6 = principle_life(x5, t);
    principle_glory(x6, t)
}

// =============== Warp Coil Structures ===============

#[derive(Clone)]
pub struct CoilSector {
    pub angle: f32,
    pub bias: ParState,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CoilType {
    Outer,
    Inner,
}

impl CoilType {
    fn label(&self) -> &'static str {
        match self {
            CoilType::Outer => "OUTER LATTICE",
            CoilType::Inner => "INNER BUBBLE",
        }
    }
}

pub struct WarpCoil {
    pub center: Vec2,
    pub radius: f32,
    pub thickness: f32,
    pub sectors: Vec<CoilSector>,
    pub params: ParacleteParams,
    pub coil_type: CoilType,
}

impl WarpCoil {
    pub fn new(center: Vec2, radius: f32, thickness: f32, num_sectors: usize, coil_type: CoilType) -> Self {
        let mut sectors = Vec::with_capacity(num_sectors);
        for s in 0..num_sectors {
            let angle = s as f32 * TAU / num_sectors as f32;
            let mut bias = [0.0f32; N_DIM];
            for i in 0..N_DIM {
                let phase = angle * (i as f32 + 1.0);
                bias[i] = 0.5 * phase.sin();
            }
            sectors.push(CoilSector { angle, bias });
        }
        Self {
            center,
            radius,
            thickness,
            sectors,
            params: ParacleteParams::default(),
            coil_type,
        }
    }

    fn sector_state(&self, sector: &CoilSector, t: f32) -> ParState {
        let mut x = [0.0f32; N_DIM];
        let p = &self.params;
        for i in 0..6 {
            let phase = sector.angle * (i as f32 + 1.0) + t * p.root_freq * TAU;
            x[i] = p.base_amp + p.drive_amp * phase.sin();
        }
        for i in 6..12 {
            let phase = sector.angle * (i as f32 + 1.0) + t * p.crown_freq * TAU;
            x[i] = p.base_amp + p.drive_amp * phase.cos();
        }
        for i in 0..N_DIM {
            x[i] += sector.bias[i];
        }
        paraclete_chain(x, t)
    }

    pub fn sample_field(&self, t: f32, pos: Vec2) -> Vec2 {
        let rel = pos - self.center;
        let r = rel.length();
        if r == 0.0 {
            return vec2(0.0, 0.0);
        }
        let dir = rel / r;
        let tang = vec2(-dir.y, dir.x);
        let mut total = vec2(0.0, 0.0);
        for sector in &self.sectors {
            let sector_dir = vec2(sector.angle.cos(), sector.angle.sin());
            let phase_offset = (dir.dot(sector_dir) * self.params.phase_speed).tanh();
            let x = self.sector_state(sector, t + phase_offset);
            let root_sum: f32 = x[0..6].iter().copied().sum();
            let crown_sum: f32 = x[6..12].iter().copied().sum();
            let within = ((r - self.radius).abs() <= self.thickness * 0.8) as i32 as f32;
            let falloff = 1.0 / (1.0 + (r - self.radius).abs() * 4.0);
            let radial_mag = within * falloff * root_sum * 0.02;
            let tang_mag = within * falloff * crown_sum * 0.02;
            total += dir * radial_mag + tang * tang_mag;
        }
        total
    }

    pub fn sample_emission(&self, t: f32, pos: Vec2) -> (f32, f32, f32, f32) {
        let rel = pos - self.center;
        let r = rel.length();
        if r == 0.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let dir = rel / r;
        let mut energy_root = 0.0;
        let mut energy_crown = 0.0;
        let mut coherence = 0.0;
        let mut weight_sum = 0.0;
        for sector in &self.sectors {
            let sector_dir = vec2(sector.angle.cos(), sector.angle.sin());
            let align = dir.dot(sector_dir).max(0.0);
            if align <= 0.0 {
                continue;
            }
            let dist_shell = (r - self.radius).abs();
            let radial_window = (-dist_shell * 4.0).exp();
            if radial_window < 1e-3 {
                continue;
            }
            let phase_offset = self.params.phase_speed * align;
            let x = self.sector_state(sector, t + phase_offset);
            let root_sum: f32 = x[0..6].iter().copied().sum();
            let crown_sum: f32 = x[6..12].iter().copied().sum();
            let root_energy = x[0..6].iter().map(|v| v * v).sum::<f32>();
            let crown_energy = x[6..12].iter().map(|v| v * v).sum::<f32>();
            let local_energy = root_energy + crown_energy;
            let local_coherence = (root_sum - crown_sum).abs() / (1.0 + local_energy);
            let w = align * radial_window;
            energy_root += root_energy * w;
            energy_crown += crown_energy * w;
            coherence += local_coherence * w;
            weight_sum += w;
        }
        if weight_sum > 0.0 {
            energy_root /= weight_sum;
            energy_crown /= weight_sum;
            coherence /= weight_sum;
        }
        let local_total = energy_root + energy_crown;
        let base_osc = (t * self.params.light_freq * TAU).sin().abs();
        let light_amp = local_total * (0.5 + 0.5 * base_osc);
        let harmonic_signal = (energy_root.sqrt() - energy_crown.sqrt()).abs();
        let resonance_delta = (harmonic_signal - self.params.resonance_target).abs();
        let resonance = (1.0 - (resonance_delta / self.params.resonance_width)).max(0.0).min(1.0);
        let whiteness = (coherence * 0.5 + resonance * 0.5).min(1.0);
        (light_amp, resonance, coherence, whiteness)
    }

    pub fn draw_coil(&self, t: f32) {
        let segments = 192;
        let is_inner = self.coil_type == CoilType::Inner;
        let base_alpha = if is_inner { 180 } else { 255 };
        let base_gray = if is_inner { 12 } else { 6 };
        
        for s in 0..segments {
            let a0 = s as f32 * TAU / segments as f32;
            let a1 = (s + 1) as f32 * TAU / segments as f32;
            let p0_outer = self.center + vec2(a0.cos(), a0.sin()) * (self.radius + self.thickness * 0.5);
            let p1_outer = self.center + vec2(a1.cos(), a1.sin()) * (self.radius + self.thickness * 0.5);
            let p0_inner = self.center + vec2(a0.cos(), a0.sin()) * (self.radius - self.thickness * 0.5);
            let p1_inner = self.center + vec2(a1.cos(), a1.sin()) * (self.radius - self.thickness * 0.5);
            let fill_color = Color::from_rgba(base_gray, base_gray, base_gray + 4, base_alpha);
            draw_triangle(p0_inner, p0_outer, p1_outer, fill_color);
            draw_triangle(p0_inner, p1_outer, p1_inner, fill_color);
        }
        
        for (idx, sector) in self.sectors.iter().enumerate() {
            let a = sector.angle;
            let inner = self.center + vec2(a.cos(), a.sin()) * (self.radius - self.thickness * 0.55);
            let outer = self.center + vec2(a.cos(), a.sin()) * (self.radius + self.thickness * 0.55);
            let phase = (t * 0.7 + idx as f32 * 0.21).sin().abs();
            
            let color = if is_inner {
                match idx % 6 {
                    0 => Color::from_rgba((180.0 * phase) as u8, 60, (120.0 * phase) as u8, 200),
                    1 => Color::from_rgba(60, (180.0 * phase) as u8, (180.0 * phase) as u8, 200),
                    2 => Color::from_rgba((100.0 * phase) as u8, (200.0 * phase) as u8, (200.0 * phase) as u8, 200),
                    3 => Color::from_rgba(40, (180.0 * phase) as u8, (220.0 * phase) as u8, 200),
                    4 => Color::from_rgba((80.0 * phase) as u8, (100.0 * phase) as u8, (220.0 * phase) as u8, 200),
                    _ => Color::from_rgba((140.0 * phase) as u8, (80.0 * phase) as u8, (200.0 * phase) as u8, 200),
                }
            } else {
                match idx % 6 {
                    0 => Color::from_rgba((255.0 * phase) as u8, 32, 32, 255),
                    1 => Color::from_rgba((255.0 * phase) as u8, (160.0 * phase) as u8, 20, 255),
                    2 => Color::from_rgba((220.0 * phase) as u8, (220.0 * phase) as u8, 16, 255),
                    3 => Color::from_rgba(20, (220.0 * phase) as u8, 60, 255),
                    4 => Color::from_rgba(24, 80, (240.0 * phase) as u8, 255),
                    _ => Color::from_rgba((160.0 * phase) as u8, 40, (255.0 * phase) as u8, 255),
                }
            };
            draw_line(inner.x, inner.y, outer.x, outer.y, 3.0, color);
        }
        
        if !is_inner {
            let null_r = self.radius * 0.2;
            let null_color = Color::from_rgba(4, 4, 4, 255);
            draw_circle(self.center.x, self.center.y, null_r, null_color);
            let glow = (t * 4.3).sin().abs();
            let edge_color = Color::from_rgba(
                (220.0 * glow) as u8,
                (220.0 * glow) as u8,
                (255.0 * glow) as u8,
                220,
            );
            draw_circle_lines(self.center.x, self.center.y, null_r * 1.02, 2.0, edge_color);
        }
    }
}

// =============== Warp Bubble ===============

pub struct WarpBubble {
    pub outer_coil: WarpCoil,
    pub inner_coil: WarpCoil,
    pub shell_params: BubbleShellParams,
    pub metrics: BubbleMetrics,
}

impl WarpBubble {
    pub fn new(center: Vec2, outer_radius: f32, outer_thickness: f32) -> Self {
        let shell_params = BubbleShellParams::default();
        let inner_radius = outer_radius * shell_params.inner_radius_ratio;
        let inner_thickness = outer_thickness * 0.6;
        
        Self {
            outer_coil: WarpCoil::new(center, outer_radius, outer_thickness, 12, CoilType::Outer),
            inner_coil: WarpCoil::new(center, inner_radius, inner_thickness, 8, CoilType::Inner),
            shell_params,
            metrics: BubbleMetrics::default(),
        }
    }
    
    pub fn update_geometry(&mut self, center: Vec2, outer_radius: f32, outer_thickness: f32) {
        self.outer_coil.center = center;
        self.outer_coil.radius = outer_radius;
        self.outer_coil.thickness = outer_thickness;
        
        let inner_radius = outer_radius * self.shell_params.inner_radius_ratio;
        let inner_thickness = outer_thickness * 0.6;
        self.inner_coil.center = center;
        self.inner_coil.radius = inner_radius;
        self.inner_coil.thickness = inner_thickness;
    }
    
    pub fn sample_combined_field(&self, t: f32, pos: Vec2) -> Vec2 {
        let outer_field = self.outer_coil.sample_field(t, pos);
        let inner_field = self.inner_coil.sample_field(t + self.shell_params.phase_offset, pos);
        let coupling = self.shell_params.coupling_strength;
        outer_field * (1.0 - coupling * 0.5) + inner_field * (1.0 + coupling * 0.5)
    }
    
    pub fn sample_combined_emission(&self, t: f32, pos: Vec2) -> (f32, f32, f32, f32) {
        let (la1, r1, c1, w1) = self.outer_coil.sample_emission(t, pos);
        let (la2, r2, c2, w2) = self.inner_coil.sample_emission(t + self.shell_params.phase_offset, pos);
        let softness = self.shell_params.shell_softness;
        (
            la1 + la2 * softness,
            (r1 + r2 * softness) / (1.0 + softness),
            (c1 + c2 * softness) / (1.0 + softness),
            (w1 + w2 * softness) / (1.0 + softness),
        )
    }
    
    pub fn compute_metrics(&mut self, t: f32) {
        let center = self.outer_coil.center;
        let null_radius = self.outer_coil.radius * 0.15;
        let sample_count = 16;
        
        let mut inner_magnitudes = Vec::with_capacity(sample_count);
        let mut gradient_samples = Vec::with_capacity(sample_count);
        
        for i in 0..sample_count {
            let angle = i as f32 * TAU / sample_count as f32;
            let dir = vec2(angle.cos(), angle.sin());
            
            let inner_pos = center + dir * null_radius;
            let inner_field = self.sample_combined_field(t, inner_pos);
            inner_magnitudes.push(inner_field.length());
            
            let outer_pos = center + dir * (null_radius * 2.5);
            let outer_field = self.sample_combined_field(t, outer_pos);
            
            gradient_samples.push((outer_field.length() - inner_field.length()).abs());
        }
        
        let inner_mean: f32 = inner_magnitudes.iter().sum::<f32>() / sample_count as f32;
        let inner_variance: f32 = inner_magnitudes.iter()
            .map(|m| (m - inner_mean).powi(2))
            .sum::<f32>() / sample_count as f32;
        self.metrics.flatness = 1.0 / (1.0 + inner_variance * 10.0);
        
        let gradient_mean: f32 = gradient_samples.iter().sum::<f32>() / sample_count as f32;
        self.metrics.shear_index = (gradient_mean * 2.0).min(1.0);
        
        let (_, r_outer, _, _) = self.outer_coil.sample_emission(t, center);
        let (_, r_inner, _, _) = self.inner_coil.sample_emission(t, center);
        self.metrics.coherence_ratio = 1.0 - (r_outer - r_inner).abs().min(1.0);
        
        let null_energy: f32 = inner_magnitudes.iter().sum::<f32>() / sample_count as f32;
        self.metrics.null_stability = (1.0 - null_energy * 5.0).max(0.0).min(1.0);
    }
    
    pub fn draw(&self, t: f32, show_inner: bool, show_outer: bool) {
        if show_inner {
            self.inner_coil.draw_coil(t + self.shell_params.phase_offset);
        }
        if show_outer {
            self.outer_coil.draw_coil(t);
        }
    }
}

// =============== Field Visualization ===============

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FieldViewMode {
    Emission,
    Contour,
}

pub struct FieldVisualizer {
    pub contour_levels: usize,
    pub cross_section_angle: f32,
}

impl Default for FieldVisualizer {
    fn default() -> Self {
        Self {
            contour_levels: 12,
            cross_section_angle: 0.0,
        }
    }
}

impl FieldVisualizer {
    pub fn draw_contour_field(&self, bubble: &WarpBubble, t: f32, w: f32, h: f32) {
        let step = 6.0;
        let center = bubble.outer_coil.center;
        let max_r = bubble.outer_coil.radius * 1.8;
        
        // Sample field magnitudes to find range
        let mut max_mag: f32 = 0.001;
        let mut y = center.y - max_r;
        while y < center.y + max_r {
            let mut x = center.x - max_r;
            while x < center.x + max_r {
                let pos = vec2(x, y);
                if (pos - center).length() < max_r {
                    let mag = bubble.sample_combined_field(t, pos).length();
                    if mag > max_mag { max_mag = mag; }
                }
                x += step * 2.0;
            }
            y += step * 2.0;
        }
        
        // Draw contour-colored field
        let mut y = 0.0;
        while y < h {
            let mut x = 0.0;
            while x < w {
                let pos = vec2(x + step * 0.5, y + step * 0.5);
                let dist = (pos - center).length();
                if dist < max_r * 1.2 {
                    let mag = bubble.sample_combined_field(t, pos).length();
                    let norm = (mag / max_mag).min(1.0);
                    
                    // Quantize to contour levels
                    let level = (norm * self.contour_levels as f32) as usize;
                    let level_norm = level as f32 / self.contour_levels as f32;
                    
                    // Color based on level (blue -> cyan -> green -> yellow -> red)
                    let (r, g, b) = if level_norm < 0.25 {
                        let t = level_norm * 4.0;
                        (0.0, t, 1.0)
                    } else if level_norm < 0.5 {
                        let t = (level_norm - 0.25) * 4.0;
                        (0.0, 1.0, 1.0 - t)
                    } else if level_norm < 0.75 {
                        let t = (level_norm - 0.5) * 4.0;
                        (t, 1.0, 0.0)
                    } else {
                        let t = (level_norm - 0.75) * 4.0;
                        (1.0, 1.0 - t, 0.0)
                    };
                    
                    let alpha = 0.3 + 0.5 * norm;
                    let color = Color::from_rgba(
                        (255.0 * r) as u8,
                        (255.0 * g) as u8,
                        (255.0 * b) as u8,
                        (255.0 * alpha) as u8,
                    );
                    draw_rectangle(x, y, step + 0.5, step + 0.5, color);
                }
                x += step;
            }
            y += step;
        }
        
        // Draw contour lines
        for level in 1..self.contour_levels {
            let threshold = (level as f32 / self.contour_levels as f32) * max_mag;
            self.draw_contour_line(bubble, t, center, max_r, threshold, step * 1.5);
        }
    }
    
    fn draw_contour_line(&self, bubble: &WarpBubble, t: f32, center: Vec2, max_r: f32, threshold: f32, step: f32) {
        let line_color = Color::from_rgba(255, 255, 255, 100);
        
        let mut y = center.y - max_r;
        while y < center.y + max_r {
            let mut x = center.x - max_r;
            while x < center.x + max_r {
                let pos = vec2(x, y);
                if (pos - center).length() < max_r {
                    let mag = bubble.sample_combined_field(t, pos).length();
                    let mag_right = bubble.sample_combined_field(t, pos + vec2(step, 0.0)).length();
                    let mag_down = bubble.sample_combined_field(t, pos + vec2(0.0, step)).length();
                    
                    if (mag < threshold && mag_right >= threshold) || (mag >= threshold && mag_right < threshold) {
                        draw_circle(x + step * 0.5, y, 1.5, line_color);
                    }
                    if (mag < threshold && mag_down >= threshold) || (mag >= threshold && mag_down < threshold) {
                        draw_circle(x, y + step * 0.5, 1.5, line_color);
                    }
                }
                x += step;
            }
            y += step;
        }
    }
    
    pub fn draw_cross_section(&self, bubble: &WarpBubble, t: f32, panel_x: f32, panel_y: f32, panel_w: f32, panel_h: f32) {
        let center = bubble.outer_coil.center;
        let max_r = bubble.outer_coil.radius * 1.5;
        let samples = 100;
        
        draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::from_rgba(10, 10, 15, 240));
        draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 2.0, Color::from_rgba(60, 60, 80, 255));
        
        draw_text("FIELD CROSS-SECTION", panel_x + 10.0, panel_y + 20.0, 16.0, Color::from_rgba(200, 200, 220, 255));
        draw_text(&format!("Angle: {:.0}°", self.cross_section_angle.to_degrees()), panel_x + 10.0, panel_y + 38.0, 14.0, Color::from_rgba(160, 160, 180, 255));
        
        let dir = vec2(self.cross_section_angle.cos(), self.cross_section_angle.sin());
        let mut field_samples = Vec::with_capacity(samples);
        let mut max_field: f32 = 0.001;
        
        for i in 0..samples {
            let t_param = (i as f32 / samples as f32) * 2.0 - 1.0;
            let pos = center + dir * (t_param * max_r);
            let mag = bubble.sample_combined_field(t, pos).length();
            field_samples.push((t_param, mag));
            if mag > max_field { max_field = mag; }
        }
        
        let graph_x = panel_x + 40.0;
        let graph_y = panel_y + 50.0;
        let graph_w = panel_w - 60.0;
        let graph_h = panel_h - 80.0;
        
        draw_line(graph_x, graph_y + graph_h, graph_x + graph_w, graph_y + graph_h, 1.0, Color::from_rgba(100, 100, 120, 255));
        draw_line(graph_x + graph_w * 0.5, graph_y, graph_x + graph_w * 0.5, graph_y + graph_h, 1.0, Color::from_rgba(60, 60, 80, 255));
        
        let outer_r_norm = bubble.outer_coil.radius / max_r;
        let inner_r_norm = bubble.inner_coil.radius / max_r;
        
        let outer_x_left = graph_x + graph_w * (0.5 - outer_r_norm * 0.5);
        let outer_x_right = graph_x + graph_w * (0.5 + outer_r_norm * 0.5);
        draw_line(outer_x_left, graph_y, outer_x_left, graph_y + graph_h, 2.0, Color::from_rgba(255, 100, 50, 100));
        draw_line(outer_x_right, graph_y, outer_x_right, graph_y + graph_h, 2.0, Color::from_rgba(255, 100, 50, 100));
        
        let inner_x_left = graph_x + graph_w * (0.5 - inner_r_norm * 0.5);
        let inner_x_right = graph_x + graph_w * (0.5 + inner_r_norm * 0.5);
        draw_line(inner_x_left, graph_y, inner_x_left, graph_y + graph_h, 2.0, Color::from_rgba(100, 150, 255, 100));
        draw_line(inner_x_right, graph_y, inner_x_right, graph_y + graph_h, 2.0, Color::from_rgba(100, 150, 255, 100));
        
        for i in 1..field_samples.len() {
            let (t0, m0) = field_samples[i - 1];
            let (t1, m1) = field_samples[i];
            
            let x0 = graph_x + graph_w * (t0 * 0.5 + 0.5);
            let x1 = graph_x + graph_w * (t1 * 0.5 + 0.5);
            let y0 = graph_y + graph_h * (1.0 - m0 / max_field);
            let y1 = graph_y + graph_h * (1.0 - m1 / max_field);
            
            draw_line(x0, y0, x1, y1, 2.0, Color::from_rgba(100, 255, 150, 255));
        }
        
        draw_text("Center", graph_x + graph_w * 0.5 - 20.0, graph_y + graph_h + 15.0, 12.0, Color::from_rgba(150, 150, 170, 255));
        draw_text(&format!("{:.2}", max_field), graph_x - 35.0, graph_y + 5.0, 12.0, Color::from_rgba(150, 150, 170, 255));
        draw_text("0", graph_x - 15.0, graph_y + graph_h, 12.0, Color::from_rgba(150, 150, 170, 255));
    }
}

// =============== UI State ===============

struct UiState {
    pub selected_param: usize,
    pub show_vectors: bool,
    pub show_emission: bool,
    pub paused: bool,
    pub time_scale: f32,
    pub last_adjust_hint_time: f32,
    pub active_coil: CoilType,
    pub show_inner_coil: bool,
    pub show_outer_coil: bool,
    pub show_metrics: bool,
    pub show_physical_specs: bool,
    pub show_cross_section: bool,
    pub view_mode: FieldViewMode,
    pub base_freq_hz: f32,
    pub major_radius_cm: f32,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            selected_param: 0,
            show_vectors: true,
            show_emission: true,
            paused: false,
            time_scale: 1.0,
            last_adjust_hint_time: 0.0,
            active_coil: CoilType::Outer,
            show_inner_coil: true,
            show_outer_coil: true,
            show_metrics: true,
            show_physical_specs: true,
            show_cross_section: false,
            view_mode: FieldViewMode::Emission,
            base_freq_hz: 1000.0,
            major_radius_cm: 15.0,
        }
    }
}

struct ParamMeta<'a> {
    name: &'a str,
    min: f32,
    max: f32,
    coarse_step: f32,
    fine_step: f32,
}

fn get_param_meta() -> [ParamMeta<'static>; 8] {
    [
        ParamMeta { name: "root_freq",        min: 0.01, max: 3.0,  coarse_step: 0.10, fine_step: 0.01 },
        ParamMeta { name: "crown_freq",       min: 0.01, max: 3.0,  coarse_step: 0.10, fine_step: 0.01 },
        ParamMeta { name: "phase_speed",      min: 0.0,  max: 3.0,  coarse_step: 0.10, fine_step: 0.01 },
        ParamMeta { name: "base_amp",         min: 0.0,  max: 2.0,  coarse_step: 0.10, fine_step: 0.02 },
        ParamMeta { name: "drive_amp",        min: 0.0,  max: 2.0,  coarse_step: 0.10, fine_step: 0.02 },
        ParamMeta { name: "light_freq",       min: 1.0,  max: 60.0, coarse_step: 2.0,  fine_step: 0.5 },
        ParamMeta { name: "resonance_target", min: 0.0,  max: 4.0,  coarse_step: 0.20, fine_step: 0.05 },
        ParamMeta { name: "resonance_width",  min: 0.01, max: 1.0,  coarse_step: 0.05, fine_step: 0.01 },
    ]
}

fn get_shell_param_meta() -> [ParamMeta<'static>; 4] {
    [
        ParamMeta { name: "inner_radius_ratio", min: 0.2,  max: 0.8,  coarse_step: 0.05, fine_step: 0.01 },
        ParamMeta { name: "shell_softness",     min: 0.0,  max: 1.0,  coarse_step: 0.10, fine_step: 0.02 },
        ParamMeta { name: "coupling_strength",  min: 0.0,  max: 1.0,  coarse_step: 0.10, fine_step: 0.02 },
        ParamMeta { name: "phase_offset",       min: -2.0, max: 2.0,  coarse_step: 0.20, fine_step: 0.05 },
    ]
}

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

fn update_param(params: &mut ParacleteParams, meta: &ParamMeta, idx: usize, delta: f32) {
    match idx {
        0 => params.root_freq = clamp(params.root_freq + delta, meta.min, meta.max),
        1 => params.crown_freq = clamp(params.crown_freq + delta, meta.min, meta.max),
        2 => params.phase_speed = clamp(params.phase_speed + delta, meta.min, meta.max),
        3 => params.base_amp = clamp(params.base_amp + delta, meta.min, meta.max),
        4 => params.drive_amp = clamp(params.drive_amp + delta, meta.min, meta.max),
        5 => params.light_freq = clamp(params.light_freq + delta, meta.min, meta.max),
        6 => params.resonance_target = clamp(params.resonance_target + delta, meta.min, meta.max),
        7 => params.resonance_width = clamp(params.resonance_width + delta, meta.min, meta.max),
        _ => {}
    }
}

fn update_shell_param(params: &mut BubbleShellParams, meta: &ParamMeta, idx: usize, delta: f32) {
    match idx {
        0 => params.inner_radius_ratio = clamp(params.inner_radius_ratio + delta, meta.min, meta.max),
        1 => params.shell_softness = clamp(params.shell_softness + delta, meta.min, meta.max),
        2 => params.coupling_strength = clamp(params.coupling_strength + delta, meta.min, meta.max),
        3 => params.phase_offset = clamp(params.phase_offset + delta, meta.min, meta.max),
        _ => {}
    }
}

fn get_param_value(params: &ParacleteParams, idx: usize) -> f32 {
    match idx {
        0 => params.root_freq,
        1 => params.crown_freq,
        2 => params.phase_speed,
        3 => params.base_amp,
        4 => params.drive_amp,
        5 => params.light_freq,
        6 => params.resonance_target,
        7 => params.resonance_width,
        _ => 0.0,
    }
}

fn get_shell_param_value(params: &BubbleShellParams, idx: usize) -> f32 {
    match idx {
        0 => params.inner_radius_ratio,
        1 => params.shell_softness,
        2 => params.coupling_strength,
        3 => params.phase_offset,
        _ => 0.0,
    }
}

fn draw_control_panel(ui: &UiState, bubble: &WarpBubble, t: f32) {
    let panel_width = 400.0;
    let margin = 10.0;
    let x0 = margin;
    let y0 = margin;
    let h = screen_height() - margin * 2.0;

    draw_rectangle(x0 - 4.0, y0 - 4.0, panel_width + 8.0, h + 8.0, Color::from_rgba(8, 8, 8, 230));
    draw_rectangle(x0, y0, panel_width, h, Color::from_rgba(18, 18, 18, 240));

    let mut y = y0 + 18.0;
    draw_text("PARACLETIC WARP COIL CONTROL", x0 + 12.0, y, 20.0, Color::from_rgba(240, 240, 255, 255));
    y += 26.0;

    let coil_label = ui.active_coil.label();
    let coil_color = match ui.active_coil {
        CoilType::Outer => Color::from_rgba(255, 180, 100, 255),
        CoilType::Inner => Color::from_rgba(100, 200, 255, 255),
    };
    draw_text(&format!("EDITING: [{}]", coil_label), x0 + 12.0, y, 18.0, coil_color);
    y += 22.0;

    let status_text = if ui.paused { "TIME: PAUSED" } else { "TIME: RUNNING" };
    let status_color = if ui.paused {
        Color::from_rgba(220, 140, 140, 255)
    } else {
        Color::from_rgba(140, 220, 140, 255)
    };
    draw_text(&format!("{}  t={:.2}", status_text, t), x0 + 12.0, y, 16.0, status_color);
    y += 20.0;

    // Physical Specifications
    if ui.show_physical_specs {
        y += 4.0;
        draw_text("═══ PHYSICAL COIL SPECS ═══", x0 + 12.0, y, 16.0, Color::from_rgba(255, 220, 100, 255));
        y += 18.0;
        
        let active_params = match ui.active_coil {
            CoilType::Outer => &bubble.outer_coil.params,
            CoilType::Inner => &bubble.inner_coil.params,
        };
        let spec = PhysicalCoilSpec::from_paraclete(active_params, ui.base_freq_hz, ui.major_radius_cm);
        
        let spec_lines = [
            format!("Major Radius: {:.1} cm", spec.major_radius_cm),
            format!("Minor Radius: {:.2} cm", spec.minor_radius_cm),
            format!("Total Turns: {} ({}/sector)", spec.num_turns, spec.turns_per_sector),
            format!("Sectors: {}", spec.num_sectors),
            format!("Wire: {} AWG ({:.2} mm)", spec.wire_gauge_awg, spec.wire_diameter_mm),
            format!("Wire Length: {:.1} m", spec.winding_length_m()),
            format!("Copper Mass: {:.0} g", spec.total_copper_mass_g()),
            format!("Inductance: {:.1} µH", spec.inductance_uh),
            format!("Self-Resonance: {:.0} Hz", spec.resonant_freq_hz),
            format!("Root Freq: {:.0} Hz", spec.root_freq_hz),
            format!("Crown Freq: {:.0} Hz", spec.crown_freq_hz),
        ];
        
        for line in &spec_lines {
            draw_text(line, x0 + 16.0, y, 13.0, Color::from_rgba(220, 200, 150, 255));
            y += 15.0;
        }
        y += 6.0;
    }

    // Bubble Metrics
    if ui.show_metrics {
        draw_text("═══ BUBBLE METRICS ═══", x0 + 12.0, y, 16.0, Color::from_rgba(150, 220, 255, 255));
        y += 18.0;
        
        let metrics = &bubble.metrics;
        let metric_items = [
            ("Flatness", metrics.flatness, Color::from_rgba(100, 200, 100, 255)),
            ("Shear", metrics.shear_index, Color::from_rgba(255, 150, 100, 255)),
            ("Coherence", metrics.coherence_ratio, Color::from_rgba(100, 150, 255, 255)),
            ("Null Stability", metrics.null_stability, Color::from_rgba(200, 200, 255, 255)),
        ];
        
        for (name, value, color) in metric_items.iter() {
            let bar_x = x0 + 110.0;
            let bar_y = y - 10.0;
            let bar_w = 200.0;
            let bar_h = 12.0;
            
            draw_rectangle(bar_x, bar_y, bar_w, bar_h, Color::from_rgba(30, 30, 40, 255));
            draw_rectangle(bar_x, bar_y, bar_w * value.clamp(0.0, 1.0), bar_h, *color);
            
            draw_text(&format!("{}: {:.3}", name, value), x0 + 12.0, y, 13.0, Color::from_rgba(200, 200, 220, 255));
            y += 16.0;
        }
        y += 6.0;
    }

    // Parameters
    draw_text("═══ PARACLETE PARAMS [1-8] ═══", x0 + 12.0, y, 14.0, Color::from_rgba(180, 180, 200, 255));
    y += 16.0;

    let active_params = match ui.active_coil {
        CoilType::Outer => &bubble.outer_coil.params,
        CoilType::Inner => &bubble.inner_coil.params,
    };

    let meta = get_param_meta();
    for (idx, m) in meta.iter().enumerate() {
        let val = get_param_value(active_params, idx);
        let is_selected = ui.selected_param == idx;
        let bar_x = x0 + 18.0;
        let bar_y = y + 2.0;
        let bar_w = panel_width - 36.0;
        let bar_h = 10.0;
        let norm = if m.max > m.min { (val - m.min) / (m.max - m.min) } else { 0.0 };
        
        let bg_col = if is_selected { Color::from_rgba(40, 40, 60, 255) } else { Color::from_rgba(25, 25, 35, 255) };
        draw_rectangle(bar_x, bar_y, bar_w, bar_h, bg_col);
        let fill_col = if is_selected { Color::from_rgba(120, 200, 255, 255) } else { Color::from_rgba(70, 120, 180, 255) };
        draw_rectangle(bar_x, bar_y, bar_w * norm.clamp(0.0, 1.0), bar_h, fill_col);

        let col = if is_selected { Color::from_rgba(230, 240, 255, 255) } else { Color::from_rgba(170, 170, 190, 255) };
        draw_text(&format!("{}.{:14}={:.3}", idx + 1, m.name, val), x0 + 12.0, y, 13.0, col);
        y += 18.0;
    }

    // Shell params
    y += 4.0;
    draw_text("SHELL [9,0,-,=]:", x0 + 12.0, y, 14.0, Color::from_rgba(180, 180, 200, 255));
    y += 16.0;

    let shell_meta = get_shell_param_meta();
    for (idx, m) in shell_meta.iter().enumerate() {
        let val = get_shell_param_value(&bubble.shell_params, idx);
        let is_selected = ui.selected_param == 8 + idx;
        let col = if is_selected { Color::from_rgba(240, 220, 255, 255) } else { Color::from_rgba(170, 160, 180, 255) };
        draw_text(&format!("{:16}={:.3}", m.name, val), x0 + 16.0, y, 12.0, col);
        y += 14.0;
    }

    // Controls help
    y += 8.0;
    draw_text("═══ CONTROLS ═══", x0 + 12.0, y, 14.0, Color::from_rgba(180, 180, 200, 255));
    y += 16.0;
    
    let help_lines = [
        "SPACE:pause  C:coil  X:cross-section",
        "1-8:param  ↑↓←→:adjust  R:reset",
        "G:contour  V:vectors  B:emission",
        "S:specs  M:metrics  I/O:inner/outer",
        "Z/.:rotate cross-section angle",
    ];
    for line in &help_lines {
        draw_text(line, x0 + 12.0, y, 12.0, Color::from_rgba(150, 150, 170, 255));
        y += 14.0;
    }
}

// =============== Main ===============

fn window_conf() -> Conf {
    Conf {
        window_title: "Paracletic Warp Coil - Physical Design Tool".to_string(),
        window_width: 1400,
        window_height: 800,
        high_dpi: true,
        fullscreen: false,
        sample_count: 4,
        window_resizable: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut bubble = WarpBubble::new(vec2(700.0, 400.0), 260.0, 54.0);
    let mut ui = UiState::default();
    let mut field_viz = FieldVisualizer::default();
    let mut t: f32 = 0.0;
    let mut resonance_focus = false;

    loop {
        let dt = get_frame_time();
        let raw_time = get_time() as f32;

        if !ui.paused {
            t += dt * ui.time_scale;
        }

        bubble.compute_metrics(t);

        // Input
        if is_key_pressed(KeyCode::Space) { ui.paused = !ui.paused; }
        if is_key_pressed(KeyCode::V) { ui.show_vectors = !ui.show_vectors; }
        if is_key_pressed(KeyCode::B) { ui.show_emission = !ui.show_emission; }
        if is_key_pressed(KeyCode::M) { ui.show_metrics = !ui.show_metrics; }
        if is_key_pressed(KeyCode::S) { ui.show_physical_specs = !ui.show_physical_specs; }
        if is_key_pressed(KeyCode::X) { ui.show_cross_section = !ui.show_cross_section; }
        if is_key_pressed(KeyCode::G) {
            ui.view_mode = if ui.view_mode == FieldViewMode::Contour {
                FieldViewMode::Emission
            } else {
                FieldViewMode::Contour
            };
        }
        if is_key_pressed(KeyCode::R) {
            bubble.outer_coil.params = ParacleteParams::default();
            bubble.inner_coil.params = ParacleteParams::default();
            bubble.shell_params = BubbleShellParams::default();
        }
        if is_key_pressed(KeyCode::C) {
            ui.active_coil = match ui.active_coil {
                CoilType::Outer => CoilType::Inner,
                CoilType::Inner => CoilType::Outer,
            };
        }
        if is_key_pressed(KeyCode::I) { ui.show_inner_coil = !ui.show_inner_coil; }
        if is_key_pressed(KeyCode::O) { ui.show_outer_coil = !ui.show_outer_coil; }
        if is_key_pressed(KeyCode::Tab) { ui.selected_param = (ui.selected_param + 1) % 12; }
        
        if is_key_pressed(KeyCode::Key1) { ui.selected_param = 0; }
        if is_key_pressed(KeyCode::Key2) { ui.selected_param = 1; }
        if is_key_pressed(KeyCode::Key3) { ui.selected_param = 2; }
        if is_key_pressed(KeyCode::Key4) { ui.selected_param = 3; }
        if is_key_pressed(KeyCode::Key5) { ui.selected_param = 4; }
        if is_key_pressed(KeyCode::Key6) { ui.selected_param = 5; }
        if is_key_pressed(KeyCode::Key7) { ui.selected_param = 6; }
        if is_key_pressed(KeyCode::Key8) { ui.selected_param = 7; }
        if is_key_pressed(KeyCode::Key9) { ui.selected_param = 8; }
        if is_key_pressed(KeyCode::Key0) { ui.selected_param = 9; }
        if is_key_pressed(KeyCode::Minus) { ui.selected_param = 10; }
        if is_key_pressed(KeyCode::Equal) { ui.selected_param = 11; }
        
        if is_key_pressed(KeyCode::Q) { ui.time_scale = (ui.time_scale - 0.1).max(0.0); }
        if is_key_pressed(KeyCode::E) { ui.time_scale = (ui.time_scale + 0.1).min(10.0); }
        if is_key_pressed(KeyCode::F) { resonance_focus = !resonance_focus; }
        
        // Cross-section angle
        if is_key_down(KeyCode::Z) { field_viz.cross_section_angle -= 0.02; }
        if is_key_down(KeyCode::Period) { field_viz.cross_section_angle += 0.02; }

        // Parameter adjustment
        let idx = ui.selected_param;
        let mut adjusted = false;
        
        if idx < 8 {
            let meta = get_param_meta();
            let m = &meta[idx];
            let active_params = match ui.active_coil {
                CoilType::Outer => &mut bubble.outer_coil.params,
                CoilType::Inner => &mut bubble.inner_coil.params,
            };
            
            if is_key_down(KeyCode::Up) { update_param(active_params, m, idx, m.fine_step); adjusted = true; }
            if is_key_down(KeyCode::Down) { update_param(active_params, m, idx, -m.fine_step); adjusted = true; }
            if is_key_down(KeyCode::Right) { update_param(active_params, m, idx, m.coarse_step); adjusted = true; }
            if is_key_down(KeyCode::Left) { update_param(active_params, m, idx, -m.coarse_step); adjusted = true; }
        } else {
            let shell_idx = idx - 8;
            let shell_meta = get_shell_param_meta();
            if shell_idx < shell_meta.len() {
                let m = &shell_meta[shell_idx];
                if is_key_down(KeyCode::Up) { update_shell_param(&mut bubble.shell_params, m, shell_idx, m.fine_step); adjusted = true; }
                if is_key_down(KeyCode::Down) { update_shell_param(&mut bubble.shell_params, m, shell_idx, -m.fine_step); adjusted = true; }
                if is_key_down(KeyCode::Right) { update_shell_param(&mut bubble.shell_params, m, shell_idx, m.coarse_step); adjusted = true; }
                if is_key_down(KeyCode::Left) { update_shell_param(&mut bubble.shell_params, m, shell_idx, -m.coarse_step); adjusted = true; }
            }
        }
        
        if adjusted { ui.last_adjust_hint_time = raw_time; }

        clear_background(Color::from_rgba(0, 0, 0, 255));

        let (w, h) = (screen_width(), screen_height());
        let center = vec2(w * 0.58, h * 0.5);
        let base_radius = h.min(w) * 0.25;
        let base_thickness = base_radius * 0.22;
        bubble.update_geometry(center, base_radius, base_thickness);

        // Draw field
        if ui.view_mode == FieldViewMode::Contour {
            field_viz.draw_contour_field(&bubble, t, w, h);
        } else if ui.show_emission {
            let step = 8.0;
            let mut fy = 0.0;
            while fy < h {
                let mut fx = 0.0;
                while fx < w {
                    let p = vec2(fx + step * 0.5, fy + step * 0.5);
                    let (light_amp, resonance, coherence, whiteness) = bubble.sample_combined_emission(t, p);
                    if light_amp > 0.001 {
                        if resonance_focus && whiteness < 0.7 {
                            fx += step;
                            continue;
                        }
                        let base_intensity = (light_amp * 0.9).min(3.0);
                        let norm_intensity = (base_intensity / 3.0).min(1.0);
                        let hue_mix = coherence.max(resonance).min(1.0);
                        let r_base = norm_intensity;
                        let g_base = norm_intensity * (0.2 + 0.8 * hue_mix);
                        let b_base = norm_intensity * (0.3 + 0.7 * (1.0 - hue_mix));
                        let white_factor = whiteness.min(1.0);
                        let r = (r_base * (1.0 - white_factor) + white_factor).min(1.0);
                        let g = (g_base * (1.0 - white_factor) + white_factor).min(1.0);
                        let b = (b_base * (1.0 - white_factor) + white_factor).min(1.0);
                        let alpha = (norm_intensity * 0.9 + resonance * 0.4).min(1.0);
                        let color = Color::from_rgba(
                            (255.0 * r) as u8,
                            (255.0 * g) as u8,
                            (255.0 * b) as u8,
                            (255.0 * alpha) as u8,
                        );
                        draw_rectangle(fx, fy, step + 0.5, step + 0.5, color);
                    }
                    fx += step;
                }
                fy += step;
            }
        }

        // Draw vectors
        if ui.show_vectors {
            let field_step = 28.0;
            let mut fy = field_step * 1.2;
            while fy < h - field_step * 1.2 {
                let mut fx = field_step * 1.2 + w * 0.22;
                while fx < w - field_step * 1.2 {
                    let p = vec2(fx, fy);
                    let f = bubble.sample_combined_field(t, p);
                    let mag = f.length();
                    let max_mag = 1.2;
                    let n = (mag / max_mag).min(1.0);
                    if n > 0.01 {
                        let dir = f.normalize_or_zero();
                        let len = 12.0 + 20.0 * n;
                        let q = p + dir * len;
                        let brightness = (180.0 * n) as u8;
                        let color = Color::from_rgba(
                            brightness,
                            (brightness as f32 * 0.6) as u8,
                            (brightness as f32 * 0.3) as u8,
                            200,
                        );
                        draw_line(p.x, p.y, q.x, q.y, 1.0 + n, color);
                    }
                    fx += field_step;
                }
                fy += field_step;
            }
        }

        bubble.draw(t, ui.show_inner_coil, ui.show_outer_coil);
        
        // Draw cross-section panel
        if ui.show_cross_section {
            let panel_w = 350.0;
            let panel_h = 200.0;
            let panel_x = w - panel_w - 20.0;
            let panel_y = h - panel_h - 20.0;
            field_viz.draw_cross_section(&bubble, t, panel_x, panel_y, panel_w, panel_h);
            
            // Draw cross-section line on main view
            let line_dir = vec2(field_viz.cross_section_angle.cos(), field_viz.cross_section_angle.sin());
            let line_len = base_radius * 1.5;
            let p1 = center - line_dir * line_len;
            let p2 = center + line_dir * line_len;
            draw_line(p1.x, p1.y, p2.x, p2.y, 2.0, Color::from_rgba(100, 255, 150, 150));
        }
        
        draw_control_panel(&ui, &bubble, t);

        next_frame().await;
    }
}
