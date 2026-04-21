-- Add Statcast horizontal / induced vertical break (inches) and spin axis (degrees).
-- Run after schema exists; safe on re-run (IF NOT EXISTS).

ALTER TABLE statcast_pitches_2023 ADD COLUMN IF NOT EXISTS pfx_x REAL;
ALTER TABLE statcast_pitches_2023 ADD COLUMN IF NOT EXISTS pfx_z REAL;
ALTER TABLE statcast_pitches_2023 ADD COLUMN IF NOT EXISTS spin_axis REAL;

ALTER TABLE statcast_pitches_2024 ADD COLUMN IF NOT EXISTS pfx_x REAL;
ALTER TABLE statcast_pitches_2024 ADD COLUMN IF NOT EXISTS pfx_z REAL;
ALTER TABLE statcast_pitches_2024 ADD COLUMN IF NOT EXISTS spin_axis REAL;

ALTER TABLE statcast_pitches_2025 ADD COLUMN IF NOT EXISTS pfx_x REAL;
ALTER TABLE statcast_pitches_2025 ADD COLUMN IF NOT EXISTS pfx_z REAL;
ALTER TABLE statcast_pitches_2025 ADD COLUMN IF NOT EXISTS spin_axis REAL;
