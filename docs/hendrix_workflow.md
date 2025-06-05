graph TD
    subgraph "Hendrix's Workflow (Data & EDA)"
        subgraph "Week 1: Data Foundation & Initial Processing"
            H_W1_T1[Day 1-2: Project Setup & Environment Config]
            H_W1_T2[Day 1-2: Load Data & Conduct Initial EDA (.shape, .info, .describe)]
            H_W1_T3[Day 1-2: CRUCIAL: Confirm WQI Relationship & Report Findings to Team]
            H_W1_T4[Day 1-2: Detailed EDA (Target/Feature Dist., Identifiers, Categorical/Numerical Assess.)]
            H_W1_T5[Day 3-4: Data Preprocessing (Feature Selection - WQI/ID Excl., Lat/Lon decisions)]
            H_W1_T6[Day 3-4: Missing Value Imputation & Outlier Treatment]
            H_W1_T7[Day 4 End: Handover Preprocessed Dataset v1 to Fernado]

            H_W1_T1 --> H_W1_T2;
            H_W1_T2 --> H_W1_T3;
            H_W1_T2 --> H_W1_T4;
            H_W1_T4 --> H_W1_T5;
            H_W1_T5 --> H_W1_T6;
            H_W1_T6 --> H_W1_T7;
        end

        subgraph "Week 2: Finalization & Documentation"
            H_W2_T1[Day 8-10: Finalize 'Dataset Description, EDA, & Preprocessing Steps' Report Section]
            H_W2_T2[Day 8-10: Prepare Data-Related Visuals for Video Showcase]
            H_W2_T3[Provide Finalized Section to Ryan for Report Compilation]

            H_W1_T6 --> H_W2_T1; % Based on completed preprocessing
            H_W2_T1 --> H_W2_T3;
            H_W2_T1 --> H_W2_T2;
        end
    end