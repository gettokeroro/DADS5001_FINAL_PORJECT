-- DDL for HOSPITALS table
-- Source: data/processed/hospitals_thailand.csv (1581 rows · 77 provinces)
-- Self-contained: idempotent (CREATE IF NOT EXISTS) — safe to re-run
--
-- Column types rationale:
--   h_code      : VARCHAR (numeric-looking but treated as code, may have leading zeros in future)
--   beds        : FLOAT (decimal like 524.0 · 1 row has empty beds → NULL allowed by default)
--   health_region : INT (values 1-13)
--   specialty_note : VARCHAR(2000) (max observed = 1141 chars · headroom for future edits)

CREATE DATABASE IF NOT EXISTS DADS5001;
CREATE SCHEMA IF NOT EXISTS DADS5001.TRIAGE;
USE DATABASE DADS5001;
USE SCHEMA TRIAGE;

CREATE OR REPLACE TABLE HOSPITALS (
    province        VARCHAR(50),
    hospital_th     VARCHAR(300),
    hospital_en     VARCHAR(300),
    h_code          VARCHAR(20),
    affiliation     VARCHAR(200),
    hospital_type   VARCHAR(100),
    beds            FLOAT,
    specialty_note  VARCHAR(2000),
    health_region   INT
);

-- Verify
DESC TABLE HOSPITALS;
