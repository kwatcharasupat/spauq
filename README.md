# bss_spatial_eval

## Development Plan

- Get bss_eval v3. **Note: v4 DOES NOT define spatial ISR**
- Datasets: MUSDB18 (for music), TIMIT (speech), SiSEC Test Set (speech)
- Spatialization: pyroomacoustics
  - Source-to-mic distance: 1m, 5m
  - Mic: Cardoid, Figure 8
  - RT60: 150ms, 300ms, 600ms (follow SiSEC)
  - Room Dimension: 3x3, 5x5, 10x10, 20x20 m^2
  - Angle: 0 to 360, 10 deg step
  - Error: +0 to +360, 10 deg step
