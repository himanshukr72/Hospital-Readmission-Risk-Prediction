# Stage 1 findings
- 101,766 encounters, 50 original columns, 71,518 unique patients.
- No duplicate records or encounter IDs after whitespace/missing-marker normalization.
- 16,773 patients have multiple encounters: group splits by patient are required.
- Early readmission: 11,357 (11.16%); other outcomes: 90,409.
- Missingness: weight 96.86%, medical specialty 49.08%, payer code 39.56%, race 2.23%.
- Literal lab-result `None` retained; `?` and blank values normalized.
- Proposed death/hospice exclusions (IDs 11, 13, 14, 19, 20, 21): 2,423 encounters, including 43 positives. Not applied pending cohort definition.
- No imputation, fitted encoding, balancing or model training performed.
