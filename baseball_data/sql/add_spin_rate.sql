-- Add release_spin_rate column to statcast_pitches_2025 table
-- Run this to add the column to an existing table without dropping data

ALTER TABLE statcast_pitches_2025 ADD COLUMN IF NOT EXISTS release_spin_rate REAL;

