-- DDL for DISEASE_DRUG table
-- Source: data/processed/disease_drug_mapping_v2_ed.csv (109 rows · 48 diseases)
-- Self-contained: idempotent (CREATE IF NOT EXISTS) — safe to re-run

CREATE DATABASE IF NOT EXISTS DADS5001;
CREATE SCHEMA IF NOT EXISTS DADS5001.TRIAGE;
USE DATABASE DADS5001;
USE SCHEMA TRIAGE;

CREATE OR REPLACE TABLE DISEASE_DRUG (
    disease_en           VARCHAR(100),
    disease_th           VARCHAR(100),
    drug_generic         VARCHAR(100),
    drug_th              VARCHAR(200),
    ed_category          VARCHAR(10),
    ed_category_meaning  VARCHAR(200),
    reimbursement_note   VARCHAR(500),
    indication_th        VARCHAR(500),
    dose_note            VARCHAR(500),
    prescription_tier    VARCHAR(20),
    source_xls_row       INT
);

-- Verify
DESC TABLE DISEASE_DRUG;
